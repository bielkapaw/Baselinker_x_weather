import io, re, hashlib
from datetime import datetime, date
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import requests

st.set_page_config(page_title="Baselinker x Weather", layout="wide")
PLOTLY_TMPL = "plotly_white"

# --- Minimal, professional theming (no logic changes) ---
px.defaults.template = PLOTLY_TMPL
px.defaults.height = 420

st.markdown("""
<style>
/* App width and base typography */
.stApp { background-color: #ffffff; }
.block-container { max-width: 1280px; padding-top: 1rem; padding-bottom: 2rem; }

/* Headings */
h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-weight: 600; letter-spacing: 0.2px;
}

/* Sidebar */
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
  margin-top: .25rem; margin-bottom: .5rem;
}

/* Buttons */
button[kind="secondary"], button[kind="primary"], .stButton>button {
  border-radius: 10px;
  border: 1px solid #e6e6e6;
  padding: .55rem .9rem;
  box-shadow: 0 1px 2px rgba(0,0,0,.04);
}

/* Tabs */
div[data-baseweb="tab-list"] { gap: .25rem; }
div[role="tab"] {
  border-radius: 10px 10px 0 0 !important;
  padding: .5rem .9rem !important;
  font-weight: 600;
}

/* Metrics */
[data-testid="stMetric"] { background: #fafafa; border: 1px solid #eee; border-radius: 12px; padding: .75rem; }
[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: #666; }
[data-testid="stMetricValue"] { font-weight: 700; }

/* Tables */
[data-testid="stDataFrameResizable"] {
  border: 1px solid #eee; border-radius: 12px; overflow: hidden;
}

/* Code blocks */
code, pre { font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

CITY_COORDS: Dict[str, tuple] = {
    "Warszawa": (52.2297, 21.0122),
    "Kraków": (50.0614, 19.9372),
    "Łódź": (51.7592, 19.4550),
    "Wrocław": (51.1079, 17.0385),
    "Poznań": (52.4064, 16.9252),
    "Gdańsk": (54.3520, 18.6466),
    "Gdynia": (54.5165, 18.5403),
    "Szczecin": (53.4285, 14.5528),
    "Bydgoszcz": (53.1235, 18.0084),
    "Lublin": (51.2465, 22.5684),
    "Białystok": (53.1325, 23.1688),
    "Katowice": (50.2649, 19.0238),
    "Rzeszów": (50.0413, 21.9990),
    "Olsztyn": (53.7784, 20.4801),
    "Toruń": (53.0138, 18.5984),
    "Kielce": (50.8661, 20.6286),
    "Opole": (50.67211, 17.92533),
    "Gorzów Wlkp.": (52.7368, 15.2288),
    "Zielona Góra": (51.9356, 15.5062),
}

DAILY_VARS = "temperature_2m_mean,temperature_2m_min,temperature_2m_max,precipitation_sum"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

def nearest_city(lat: float, lon: float) -> str:
    best, dmin = "Warszawa", 1e9
    for c, (la, lo) in CITY_COORDS.items():
        d = ((la - lat) ** 2 + (lo - lon) ** 2) ** 0.5
        if d < dmin:
            dmin, best = d, c
    return best

def parse_sales_csv(file_bytes: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "cp1250", "latin-1"):
        try:
            raw = file_bytes.decode(enc, errors="ignore")
            first = raw.splitlines()[0] if raw.splitlines() else ""
            sep = ";" if (";" in first or "Data dodania;" in first) else ","
            df = pd.read_csv(io.StringIO(raw), sep=sep, dtype=str)
            need = {"Data dodania", "Ilość zamówień", "Wartość zamówień"}
            if not need.issubset(df.columns):
                raise ValueError("Sales CSV: missing columns 'Data dodania','Ilość zamówień','Wartość zamówień'")
            def parse_dt(x):
                for fmt in ("%d.%m.%Y %H:%M:%S", "%d.%m.%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                    try:
                        return datetime.strptime(str(x), fmt).date()
                    except Exception:
                        pass
                d = pd.to_datetime(x, errors="coerce", dayfirst=True)
                return None if pd.isna(d) else d.date()
            df["date"] = df["Data dodania"].map(parse_dt)
            df = df.dropna(subset=["date"])
            def num(s):
                s = str(s).replace(" ", "").replace(",", ".")
                try: return float(s)
                except: return 0.0
            df["sales_count"] = df["Ilość zamówień"].map(num)
            df["sales_value"] = df["Wartość zamówień"].map(num)
            g = df.groupby("date", as_index=False).agg({"sales_count":"sum","sales_value":"sum"})
            return g.sort_values("date")
        except Exception:
            continue
    raise RuntimeError("Failed to read sales CSV (check headers/encoding).")

def parse_openmeteo_csv(file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, str]:
    text = file_bytes.decode("utf-8", errors="ignore")
    lat = lon = None
    for line in text.splitlines()[:20]:
        m1 = re.search(r"latitude\s*,\s*([\-0-9\.]+)", line, re.I)
        m2 = re.search(r"longitude\s*,\s*([\-0-9\.]+)", line, re.I)
        if m1: lat = float(m1.group(1))
        if m2: lon = float(m2.group(1))
    header_idx = None
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        t = ln.strip().lower()
        if t.startswith("time,") or t.startswith("date,"):
            header_idx = i; break
    if header_idx is None:
        raise RuntimeError(f"Missing 'time,' / 'date,' header in {filename}")
    core = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(core))
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.date
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    cols = {"temperature_2m_mean":None,"temperature_2m_min":None,"temperature_2m_max":None,"precipitation_sum":None}
    for c in df.columns:
        lc = c.strip().lower()
        if "temperature_2m_mean" in lc: cols["temperature_2m_mean"] = c
        if "temperature_2m_min" in lc:  cols["temperature_2m_min"]  = c
        if "temperature_2m_max" in lc:  cols["temperature_2m_max"]  = c
        if "precipitation_sum" in lc or lc=="precipitation": cols["precipitation_sum"] = c
    missing = [k for k,v in cols.items() if v is None]
    if missing: raise RuntimeError(f"File {filename} is missing columns: {missing}")
    out = df[["date", cols["temperature_2m_mean"], cols["temperature_2m_min"], cols["temperature_2m_max"], cols["precipitation_sum"]]].copy()
    out.columns = ["date","temperature_2m_mean","temperature_2m_min","temperature_2m_max","precipitation_sum"]
    for c in ["temperature_2m_mean","temperature_2m_min","temperature_2m_max","precipitation_sum"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    city = None
    if lat is not None and lon is not None:
        city = nearest_city(lat, lon)
    if city is None:
        low = filename.lower()
        for c in CITY_COORDS.keys():
            key = c.lower().replace("ó","o").replace("ń","n").replace("ł","l")
            if key in low: city = c; break
    return out, (city or "Warszawa")

def merge_sales_weather(sales_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(sales_df, weather_df, on="date", how="left").sort_values("date")
    for c in ["temperature_2m_mean","temperature_2m_min","temperature_2m_max","precipitation_sum"]:
        df[c] = df[c].interpolate(limit_direction="both")
    df["is_rain"] = (df["precipitation_sum"].fillna(0) > 0).astype(int)
    return df

def rolling_cols(df: pd.DataFrame, metric: str, weather: str, win: int) -> pd.DataFrame:
    d = df.copy()
    if win and win > 1:
        d[f"{metric}_ma"] = d[metric].rolling(win, min_periods=1).mean()
        d[f"{weather}_ma"] = d[weather].rolling(win, min_periods=1).mean()
        d["precipitation_ma"] = d["precipitation_sum"].rolling(win, min_periods=1).mean()
    else:
        d[f"{metric}_ma"] = d[metric]
        d[f"{weather}_ma"] = d[weather]
        d["precipitation_ma"] = d["precipitation_sum"]
    return d

def apply_lag(df: pd.DataFrame, weather: str, lag_days: int) -> pd.DataFrame:
    d = df.copy()
    if lag_days != 0:
        d[weather] = d[weather].shift(lag_days)
        if f"{weather}_ma" in d.columns:
            d[f"{weather}_ma"] = d[f"{weather}_ma"].shift(lag_days)
    return d

def dtrange(df: pd.DataFrame) -> Tuple[date, date]:
    return (df["date"].min(), df["date"].max())

def build_openmeteo_url(city: str, start: date, end: date) -> str:
    lat, lon = CITY_COORDS[city]
    return (
        f"{OPEN_METEO_ARCHIVE}"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start.isoformat()}&end_date={end.isoformat()}"
        f"&daily={DAILY_VARS}&timezone=Europe%2FWarsaw&format=csv"
    )

def hash_files(files) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.getbuffer())
    return h.hexdigest()

# --- Header ---
logo_url = "https://www.drzewa.com.pl/static/version1595339804/frontend/Pearl/weltpixel_custom/pl_PL/images/logo-konieczko.svg"
top_l, top_r = st.columns([2,4])
with top_l:
    # reduced logo size for a cleaner header
    st.image(logo_url, caption="", width=220)
with top_r:
    st.title("")  # left as-is to keep your layout intact

# --- Sidebar ---
with st.sidebar:
    st.header("Files")
    sales_file = st.file_uploader("Sales CSV (BaseLinker)", type=["csv"], key="sales_up")
    weather_files = st.file_uploader("Weather CSV (Open-Meteo) – you can select multiple", type=["csv"], accept_multiple_files=True, key="wx_up")
    st.caption('Sales: "Data dodania", "Ilość zamówień", "Wartość zamówień". Weather: "time/date", "temperature_2m_*", "precipitation_sum".')
    st.divider()
    st.header("Chart settings")
    metric = st.selectbox("Sales metric", ["sales_count","sales_value"], format_func=lambda x: "Order count" if x=="sales_count" else "Order value")
    weather_var = st.selectbox("Weather parameter (right axis)", ["temperature_2m_mean","temperature_2m_min","temperature_2m_max"], index=0,
                               format_func=lambda x: {"temperature_2m_mean":"Avg T (°C)","temperature_2m_min":"Min T (°C)","temperature_2m_max":"Max T (°C)"}[x])
    lag_days = st.slider("Weather averaging [days]", min_value=-14, max_value=14, value=0, step=1)
    ma_win = st.selectbox("Smoothing (MA)", [0,7,14], index=1)
    st.caption("Time chart: Sales (Y, left) + Temperatures (Y2, right) + Precipitation (Y3, right).")

# --- Session init ---
ss = st.session_state
ss.setdefault("sales_df", pd.DataFrame())
ss.setdefault("weather_by_city", {})
ss.setdefault("merged_df", pd.DataFrame())
ss.setdefault("cities", list(CITY_COORDS.keys()))
ss.setdefault("selected_city", "Warszawa")
ss.setdefault("wx_version", "")

# --- Load sales ---
if sales_file is not None:
    try:
        ss.sales_df = parse_sales_csv(sales_file.getvalue())
        dmin, dmax = dtrange(ss.sales_df)
        st.success(f"Sales: {len(ss.sales_df)} days ({dmin} → {dmax})")
    except Exception as e:
        st.warning(f"Sales: {e}")

# --- Load weather + auto-refresh after new upload ---
if weather_files:
    cur_hash = hash_files(weather_files)
    if cur_hash != ss.wx_version:
        ss.wx_version = cur_hash
        ss.weather_by_city = {}
        loaded = []
        for wf in weather_files:
            try:
                wx_df, city = parse_openmeteo_csv(wf.getvalue(), wf.name)
                ss.weather_by_city[city] = wx_df
                loaded.append(city)
            except Exception as e:
                st.warning(f"Weather ({wf.name}): {e}")
        if loaded:
            ss.cities = sorted(set(ss.weather_by_city.keys()))
            ss.selected_city = ss.cities[0]
            st.success("Weather: " + ", ".join(ss.cities))
            if not ss.sales_df.empty:
                ss.merged_df = merge_sales_weather(ss.sales_df, ss.weather_by_city[ss.selected_city])
                st.info("Charts updated for: " + ss.selected_city)

# --- Controls and merge ---
c1, c2 = st.columns([2,2])
city = c1.selectbox("City", ss.cities, index=(ss.cities.index(ss.selected_city) if ss.selected_city in ss.cities else 0))
update_click = c2.button("🔄 Update dataset", use_container_width=True)

if city != ss.selected_city:
    ss.selected_city = city
    if not ss.sales_df.empty and city in ss.weather_by_city:
        ss.merged_df = merge_sales_weather(ss.sales_df, ss.weather_by_city[city])
        st.info("Switched city → charts refreshed.")

if update_click:
    if ss.sales_df.empty:
        st.info("Upload the sales CSV first.")
    elif city not in ss.weather_by_city:
        st.info("Upload a weather CSV for the selected city.")
    else:
        ss.merged_df = merge_sales_weather(ss.sales_df, ss.weather_by_city[city])
        st.success(f"Updated: {len(ss.merged_df)} days | City: {city}")

if ss.merged_df.empty:
    st.info('Upload files, choose a city and click "🔄 Update dataset".')
    st.stop()

base = ss.merged_df.copy()
base["date"] = pd.to_datetime(base["date"])
all_min, all_max = base["date"].min().date(), base["date"].max().date()

# Safe date picker
try:
    picked = st.date_input("Date range (for charts)", value=(all_min, all_max), min_value=all_min, max_value=all_max)
    if isinstance(picked, tuple) and len(picked) == 2 and picked[0] and picked[1]:
        d_from, d_to = picked
    else:
        d_from, d_to = all_min, all_max
except Exception:
    d_from, d_to = all_min, all_max

mask = (base["date"].dt.date >= d_from) & (base["date"].dt.date <= d_to)
base = base.loc[mask].reset_index(drop=True)

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Days", int(base.shape[0]))
k2.metric("Orders (count)", f"{base['sales_count'].sum():.0f}")
k3.metric("Orders (value)", f"{base['sales_value'].sum():.0f} PLN")
k4.metric("Avg T (°C)", f"{base['temperature_2m_mean'].mean():.2f}")
k5.metric("Precipitation (mm)", f"{base['precipitation_sum'].sum():.2f}")

tabs = st.tabs([
    "Timeline: sales + weather",
    "Monthly",
    "Weekly pattern",
    "Heatmap",
    "Table",
    "Download CSV (PL cities)"
])

# --- TIMELINE (Y, Y2, Y3) ---
with tabs[0]:
    df = base.copy().sort_values("date")
    df = rolling_cols(df, metric, weather_var, ma_win)
    df = apply_lag(df, weather_var, lag_days)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["date"], y=df[metric], name="Sales (daily)", opacity=0.40, marker_line_width=0), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["date"], y=df[f"{metric}_ma"], mode="lines", name=f"Sales MA({ma_win})" if ma_win else "Sales MA(1)"), secondary_y=False)

    label = {"temperature_2m_mean":"Avg T", "temperature_2m_min":"Min T", "temperature_2m_max":"Max T"}[weather_var]
    fig.add_trace(go.Scatter(x=df["date"], y=df[f"{weather_var}_ma"], mode="lines", name=f"{label} (MA)"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["date"], y=df["temperature_2m_min"], mode="lines", name="Min T", line=dict(width=1, dash="dot")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["date"], y=df["temperature_2m_max"], mode="lines", name="Max T", line=dict(width=1, dash="dot")), secondary_y=True)

    fig.add_trace(go.Bar(x=df["date"], y=df["precipitation_ma"], name="Precipitation (MA)", marker_line_width=0, opacity=0.55, yaxis="y3"))

    fig.update_layout(
        template=PLOTLY_TMPL,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis3=dict(title="Precipitation (mm)", overlaying="y", side="right", position=1.0, showgrid=False),
        font=dict(size=13)
    )
    fig.update_yaxes(title_text=("Order count" if metric=="sales_count" else "Order value"), secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# --- MONTHLY ---
with tabs[1]:
    m = base.copy()
    m["month"] = m["date"].dt.to_period("M").dt.to_timestamp()
    mdf = m.groupby("month", as_index=False).agg(
        **{metric:(metric,"sum")},
        temperature_2m_mean=("temperature_2m_mean","mean"),
        precipitation_sum=("precipitation_sum","sum"),
    ).rename(columns={"month":"date"}).sort_values("date")

    figm = make_subplots(specs=[[{"secondary_y": True}]])
    figm.add_trace(go.Bar(x=mdf["date"], y=mdf[metric], name="Sales (monthly)", marker_line_width=0), secondary_y=False)
    figm.add_trace(go.Scatter(x=mdf["date"], y=mdf["temperature_2m_mean"], name="Avg T (monthly)", mode="lines+markers"), secondary_y=True)
    figm.add_trace(go.Scatter(x=mdf["date"], y=mdf["precipitation_sum"], name="Precipitation (monthly sum)", mode="lines+markers"), secondary_y=True)
    figm.update_layout(
        template=PLOTLY_TMPL,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10,r=10,t=40,b=10),
        font=dict(size=13)
    )
    figm.update_yaxes(title_text=("Count" if metric=="sales_count" else "Value"), secondary_y=False)
    figm.update_yaxes(title_text="Temp / Precipitation", secondary_y=True)
    st.plotly_chart(figm, use_container_width=True)

# --- WEEKLY PATTERN ---
with tabs[2]:
    tmp = base.copy()
    tmp["dow"] = tmp["date"].dt.dayofweek
    tmp["rain_label"] = np.where(tmp["precipitation_sum"]>0, "rain", "no rain")
    g = tmp.groupby(["dow","rain_label"], as_index=False).agg(val=(metric,"mean"))
    dow_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    g["dow_name"] = g["dow"].map(dow_map)
    figw = px.bar(
        g, x="dow_name", y="val", color="rain_label", barmode="group", template=PLOTLY_TMPL,
        labels={"dow_name":"Day of week","val":"Average sales","rain_label":"Condition"},
        title="Average sales vs day of week (rain vs no rain)"
    )
    figw.update_layout(font=dict(size=13))
    st.plotly_chart(figw, use_container_width=True)

# --- HEATMAP ---
with tabs[3]:
    h = base.copy()
    h["m"] = h["date"].dt.month
    h["d"] = h["date"].dt.day
    pivot = h.pivot_table(index="m", columns="d", values=metric, aggfunc="sum")
    figheat = px.imshow(
        pivot, aspect="auto", template=PLOTLY_TMPL, color_continuous_scale="Turbo",
        labels=dict(x="Day of month", y="Month", color=("Count" if metric=="sales_count" else "Value"))
    )
    figheat.update_yaxes(
        tickmode="array", tickvals=list(range(1,13)),
        ticktext=["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"]
    )
    figheat.update_layout(font=dict(size=13))
    st.plotly_chart(figheat, use_container_width=True)

# --- TABLE + EXPORT ---
with tabs[4]:
    st.download_button("💾 Download merged CSV", data=base.to_csv(index=False).encode("utf-8"),
                       file_name=f"merged_{ss.selected_city}.csv", mime="text/csv")
    st.dataframe(base.sort_values("date"), use_container_width=True, hide_index=True)

# --- DOWNLOAD CSV (multiple Polish cities) ---
with tabs[5]:
    st.subheader("Download weather CSV (Open-Meteo: Archive, daily)")
    col_city, col_sd, col_ed = st.columns([2,2,2])
    dl_city = col_city.selectbox("City", list(CITY_COORDS.keys()), index=list(CITY_COORDS.keys()).index("Warszawa"))
    default_start = (ss.sales_df["date"].min() if not ss.sales_df.empty else date(2024,1,1))
    default_end   = (ss.sales_df["date"].max() if not ss.sales_df.empty else date(2024,12,31))
    try:
        sd = col_sd.date_input("Start", value=default_start, min_value=date(1979,1,1), max_value=date.today())
    except Exception:
        sd = default_start
    try:
        ed = col_ed.date_input("End", value=default_end, min_value=date(1979,1,1), max_value=date.today())
    except Exception:
        ed = default_end

    if ed < sd:
        st.warning("End cannot be earlier than start.")
    url = build_openmeteo_url(dl_city, sd, ed)
    st.code(url, language="text")

    cdl1, cdl2 = st.columns([1,2])
    if cdl1.button("⬇️ Download CSV"):
        try:
            r = requests.get(url, timeout=30)
            if r.ok and r.text.strip():
                st.download_button(
                    "💾 Save file",
                    data=r.text.encode("utf-8"),
                    file_name=f"weather_{dl_city}_{sd}_{ed}.csv",
                    mime="text/csv",
                )
                st.success("Downloaded CSV from Open-Meteo.")
            else:
                st.warning(f"Failed to download (HTTP {r.status_code}).")
        except Exception as e:
            st.warning(f"Download error: {e}")

    if cdl2.button("➕ Upload directly into the app"):
        try:
            r = requests.get(url, timeout=30)
            if r.ok and r.text.strip():
                wx_df, city_infer = parse_openmeteo_csv(r.content, f"{dl_city}.csv")
                ss.weather_by_city[dl_city] = wx_df
                ss.cities = sorted(set(ss.weather_by_city.keys()))
                ss.selected_city = dl_city
                if not ss.sales_df.empty:
                    ss.merged_df = merge_sales_weather(ss.sales_df, ss.weather_by_city[dl_city])
                    st.success(f"Added weather for {dl_city} and updated charts.")
                else:
                    st.info(f"Added weather for {dl_city}. Upload sales to see charts.")
            else:
                st.warning(f"Failed to download (HTTP {r.status_code}).")
        except Exception as e:
            st.warning(f"Download error: {e}")
