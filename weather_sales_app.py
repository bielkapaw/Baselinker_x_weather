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

st.set_page_config(page_title="BielkaP 2025 x Baselinker", layout="wide")
PLOTLY_TMPL = "plotly_white"

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
                raise ValueError("CSV sprzedaży: brak kolumn 'Data dodania','Ilość zamówień','Wartość zamówień'")
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
    raise RuntimeError("Nie udało się odczytać CSV sprzedaży (sprawdź nagłówki/kodowanie).")

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
        raise RuntimeError(f"Brak nagłówka 'time,' / 'date,' w {filename}")
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
    if missing: raise RuntimeError(f"W pliku {filename} brakuje kolumn: {missing}")
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
top_l, top_r = st.columns([1,4])
with top_l:
    st.image(logo_url, caption="", width=1280)
with top_r:
    st.title("   📊 BielkaP  x Baselinker")

# --- Sidebar ---
with st.sidebar:
    st.header("Pliki")
    sales_file = st.file_uploader("CSV sprzedaży (BaseLinker)", type=["csv"], key="sales_up")
    weather_files = st.file_uploader("CSV pogody (Open-Meteo) – możesz kilka", type=["csv"], accept_multiple_files=True, key="wx_up")
    st.caption("Sprzedaż: „Data dodania”, „Ilość zamówień”, „Wartość zamówień”. Pogoda: „time/date”, „temperature_2m_*”, „precipitation_sum”.")
    st.divider()
    st.header("Ustawienia wykresu")
    metric = st.selectbox("Metryka sprzedaży", ["sales_count","sales_value"], format_func=lambda x: "Liczba zamówień" if x=="sales_count" else "Wartość zamówień")
    weather_var = st.selectbox("Parametr pogody (oś prawa)", ["temperature_2m_mean","temperature_2m_min","temperature_2m_max"], index=0,
                               format_func=lambda x: {"temperature_2m_mean":"T średnia (°C)","temperature_2m_min":"T min (°C)","temperature_2m_max":"T max (°C)"}[x])
    lag_days = st.slider("Uśrednienie pogody [dni]", min_value=-14, max_value=14, value=0, step=1)
    ma_win = st.selectbox("Wygładzanie (MA)", [0,7,14], index=1)
    st.caption("Wykres czasu: Sprzedaż (Y, lewa) + Temperatury (Y2, prawa) + Opady (Y3, prawa).")

# --- Session init ---
ss = st.session_state
ss.setdefault("sales_df", pd.DataFrame())
ss.setdefault("weather_by_city", {})
ss.setdefault("merged_df", pd.DataFrame())
ss.setdefault("cities", list(CITY_COORDS.keys()))
ss.setdefault("selected_city", "Warszawa")
ss.setdefault("wx_version", "")

# --- Wczytanie sprzedaży ---
if sales_file is not None:
    try:
        ss.sales_df = parse_sales_csv(sales_file.getvalue())
        dmin, dmax = dtrange(ss.sales_df)
        st.success(f"Sprzedaż: {len(ss.sales_df)} dni ({dmin} → {dmax})")
    except Exception as e:
        st.warning(f"Sprzedaż: {e}")

# --- Wczytanie pogody + auto-refresh po nowym uploadzie ---
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
                st.warning(f"Pogoda ({wf.name}): {e}")
        if loaded:
            ss.cities = sorted(set(ss.weather_by_city.keys()))
            ss.selected_city = ss.cities[0]
            st.success("Pogoda: " + ", ".join(ss.cities))
            if not ss.sales_df.empty:
                ss.merged_df = merge_sales_weather(ss.sales_df, ss.weather_by_city[ss.selected_city])
                st.info("Zaktualizowano wykresy dla: " + ss.selected_city)

# --- Sterowanie i scalanie ---
c1, c2 = st.columns([2,2])
city = c1.selectbox("Miasto", ss.cities, index=(ss.cities.index(ss.selected_city) if ss.selected_city in ss.cities else 0))
update_click = c2.button("🔄 Update bazy", use_container_width=True)

if city != ss.selected_city:
    ss.selected_city = city
    if not ss.sales_df.empty and city in ss.weather_by_city:
        ss.merged_df = merge_sales_weather(ss.sales_df, ss.weather_by_city[city])
        st.info("Przełączono miasto → odświeżono wykresy.")

if update_click:
    if ss.sales_df.empty:
        st.info("Wgraj najpierw CSV sprzedaży.")
    elif city not in ss.weather_by_city:
        st.info("Wgraj CSV pogody dla wybranego miasta.")
    else:
        ss.merged_df = merge_sales_weather(ss.sales_df, ss.weather_by_city[city])
        st.success(f"Zaktualizowano: {len(ss.merged_df)} dni | Miasto: {city}")

if ss.merged_df.empty:
    st.info("Wgraj pliki, wybierz miasto i kliknij „🔄 Update bazy”.")
    st.stop()

base = ss.merged_df.copy()
base["date"] = pd.to_datetime(base["date"])
all_min, all_max = base["date"].min().date(), base["date"].max().date()

# Bezpieczny date picker
try:
    picked = st.date_input("Zakres dat (dla wykresów)", value=(all_min, all_max), min_value=all_min, max_value=all_max)
    if isinstance(picked, tuple) and len(picked) == 2 and picked[0] and picked[1]:
        d_from, d_to = picked
    else:
        d_from, d_to = all_min, all_max
except Exception:
    d_from, d_to = all_min, all_max

mask = (base["date"].dt.date >= d_from) & (base["date"].dt.date <= d_to)
base = base.loc[mask].reset_index(drop=True)

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Dni", int(base.shape[0]))
k2.metric("Suma zamówień (liczba)", f"{base['sales_count'].sum():.0f}")
k3.metric("Suma zamówień (wartość)", f"{base['sales_value'].sum():.2f}")
k4.metric("Śr. T (°C)", f"{base['temperature_2m_mean'].mean():.2f}")
k5.metric("Suma opadów (mm)", f"{base['precipitation_sum'].sum():.2f}")

tabs = st.tabs([
    "Oś czasu: sprzedaż + pogoda",
    "Miesięcznie",
    "Wzorzec tygodnia",
    "Mapa cieplna",
    "Tabela",
    "Pobierz CSV (PL miasta)"
])

# --- OŚ CZASU (Y, Y2, Y3) ---
with tabs[0]:
    df = base.copy().sort_values("date")
    df = rolling_cols(df, metric, weather_var, ma_win)
    df = apply_lag(df, weather_var, lag_days)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=df["date"], y=df[metric], name="Sprzedaż (dziennie)", opacity=0.40, marker_line_width=0), secondary_y=False)
    fig.add_trace(go.Scatter(x=df["date"], y=df[f"{metric}_ma"], mode="lines", name=f"Sprzedaż MA({ma_win})" if ma_win else "Sprzedaż MA(1)"), secondary_y=False)

    label = {"temperature_2m_mean":"T śr", "temperature_2m_min":"T min", "temperature_2m_max":"T max"}[weather_var]
    fig.add_trace(go.Scatter(x=df["date"], y=df[f"{weather_var}_ma"], mode="lines", name=f"{label} (MA)"), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["date"], y=df["temperature_2m_min"], mode="lines", name="T min", line=dict(width=1, dash="dot")), secondary_y=True)
    fig.add_trace(go.Scatter(x=df["date"], y=df["temperature_2m_max"], mode="lines", name="T max", line=dict(width=1, dash="dot")), secondary_y=True)

    fig.add_trace(go.Bar(x=df["date"], y=df["precipitation_ma"], name="Opad (MA)", marker_line_width=0, opacity=0.55, yaxis="y3"))

    fig.update_layout(
        template=PLOTLY_TMPL,
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis3=dict(title="Opad (mm)", overlaying="y", side="right", position=1.0, showgrid=False)
    )
    fig.update_yaxes(title_text=("Liczba zamówień" if metric=="sales_count" else "Wartość zamówień"), secondary_y=False)
    fig.update_yaxes(title_text="Temperatura (°C)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# --- MIESIĘCZNIE ---
with tabs[1]:
    m = base.copy()
    m["month"] = m["date"].dt.to_period("M").dt.to_timestamp()
    mdf = m.groupby("month", as_index=False).agg(
        **{metric:(metric,"sum")},
        temperature_2m_mean=("temperature_2m_mean","mean"),
        precipitation_sum=("precipitation_sum","sum"),
    ).rename(columns={"month":"date"}).sort_values("date")

    figm = make_subplots(specs=[[{"secondary_y": True}]])
    figm.add_trace(go.Bar(x=mdf["date"], y=mdf[metric], name="Sprzedaż (mies.)", marker_line_width=0), secondary_y=False)
    figm.add_trace(go.Scatter(x=mdf["date"], y=mdf["temperature_2m_mean"], name="T śr (mies.)", mode="lines+markers"), secondary_y=True)
    figm.add_trace(go.Scatter(x=mdf["date"], y=mdf["precipitation_sum"], name="Opad (suma mies.)", mode="lines+markers"), secondary_y=True)
    figm.update_layout(template=PLOTLY_TMPL, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(l=10,r=10,t=40,b=10))
    figm.update_yaxes(title_text=("Liczba" if metric=="sales_count" else "Wartość"), secondary_y=False)
    figm.update_yaxes(title_text="Temp / Opad", secondary_y=True)
    st.plotly_chart(figm, use_container_width=True)

# --- WZORZEC TYGODNIA ---
with tabs[2]:
    tmp = base.copy()
    tmp["dow"] = tmp["date"].dt.dayofweek
    tmp["rain_label"] = np.where(tmp["precipitation_sum"]>0, "deszcz", "bez deszczu")
    g = tmp.groupby(["dow","rain_label"], as_index=False).agg(val=(metric,"mean"))
    dow_map = {0:"Pn",1:"Wt",2:"Śr",3:"Cz",4:"Pt",5:"So",6:"Nd"}
    g["dow_name"] = g["dow"].map(dow_map)
    figw = px.bar(g, x="dow_name", y="val", color="rain_label", barmode="group", template=PLOTLY_TMPL,
                  labels={"dow_name":"Dzień tygodnia","val":"Średnia sprzedaż","rain_label":"Warunek"},
                  title="Średnia sprzedaż vs dzień tygodnia (deszcz vs brak)")
    st.plotly_chart(figw, use_container_width=True)

# --- MAPA CIEPLNA ---
with tabs[3]:
    h = base.copy()
    h["m"] = h["date"].dt.month
    h["d"] = h["date"].dt.day
    pivot = h.pivot_table(index="m", columns="d", values=metric, aggfunc="sum")
    figheat = px.imshow(pivot, aspect="auto", template=PLOTLY_TMPL, color_continuous_scale="Turbo",
                        labels=dict(x="Dzień miesiąca", y="Miesiąc", color=("Liczba" if metric=="sales_count" else "Wartość")))
    figheat.update_yaxes(tickmode="array", tickvals=list(range(1,13)),
                         ticktext=["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"])
    st.plotly_chart(figheat, use_container_width=True)

# --- TABELA + EXPORT ---
with tabs[4]:
    st.download_button("💾 Pobierz scalony CSV", data=base.to_csv(index=False).encode("utf-8"),
                       file_name=f"merged_{ss.selected_city}.csv", mime="text/csv")
    st.dataframe(base.sort_values("date"), use_container_width=True, hide_index=True)

# --- POBIERANIE CSV (wiele polskich miast) ---
with tabs[5]:
    st.subheader("Pobierz CSV pogody (Open-Meteo: Archive, dzienne)")
    col_city, col_sd, col_ed = st.columns([2,2,2])
    dl_city = col_city.selectbox("Miasto", list(CITY_COORDS.keys()), index=list(CITY_COORDS.keys()).index("Warszawa"))
    default_start = (ss.sales_df["date"].min() if not ss.sales_df.empty else date(2024,1,1))
    default_end   = (ss.sales_df["date"].max() if not ss.sales_df.empty else date(2024,12,31))
    try:
        sd = col_sd.date_input("Start", value=default_start, min_value=date(1979,1,1), max_value=date.today())
    except Exception:
        sd = default_start
    try:
        ed = col_ed.date_input("Koniec", value=default_end, min_value=date(1979,1,1), max_value=date.today())
    except Exception:
        ed = default_end

    if ed < sd:
        st.warning("Koniec nie może być wcześniejszy niż start.")
    url = build_openmeteo_url(dl_city, sd, ed)
    st.code(url, language="text")

    cdl1, cdl2 = st.columns([1,2])
    if cdl1.button("⬇️ Pobierz CSV"):
        try:
            r = requests.get(url, timeout=30)
            if r.ok and r.text.strip():
                st.download_button(
                    "💾 Zapisz plik",
                    data=r.text.encode("utf-8"),
                    file_name=f"weather_{dl_city}_{sd}_{ed}.csv",
                    mime="text/csv",
                )
                st.success("Pobrano CSV z Open-Meteo.")
            else:
                st.warning(f"Nie udało się pobrać (HTTP {r.status_code}).")
        except Exception as e:
            st.warning(f"Błąd pobierania: {e}")

    if cdl2.button("➕ Wgraj bezpośrednio do aplikacji"):
        try:
            r = requests.get(url, timeout=30)
            if r.ok and r.text.strip():
                wx_df, city_infer = parse_openmeteo_csv(r.content, f"{dl_city}.csv")
                ss.weather_by_city[dl_city] = wx_df
                ss.cities = sorted(set(ss.weather_by_city.keys()))
                ss.selected_city = dl_city
                if not ss.sales_df.empty:
                    ss.merged_df = merge_sales_weather(ss.sales_df, ss.weather_by_city[dl_city])
                    st.success(f"Dodano pogodę dla {dl_city} i zaktualizowano wykresy.")
                else:
                    st.info(f"Dodano pogodę dla {dl_city}. Wgraj sprzedaż, aby zobaczyć wykresy.")
            else:
                st.warning(f"Nie udało się pobrać (HTTP {r.status_code}).")
        except Exception as e:
            st.warning(f"Błąd pobierania: {e}")
