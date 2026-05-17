"""
app/app.py — NYC Taxi AI · Editorial Edition
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import mlflow
import mlflow.spark
from pyspark.sql import functions as F

# ── Config ─────────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_FARE      = os.getenv("MODEL_NAME_FARE",     "xgb_fare")
MODEL_DURATION  = os.getenv("MODEL_NAME_DURATION", "xgb_duration")
DIST_MEDIA_PATH = os.getenv("DIST_MEDIA_PATH",     "/app/data/gold/dist_media")
ZONES_PATH      = os.getenv("ZONES_PATH",          "/app/data/bronze/taxi_zone_lookup.csv")
EXPERIMENT      = "nyc_taxi_model"

DAYS_FULL  = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
DAYS_SHORT = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MONTHS     = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]

# Spark SQL dayofweek convention: 1=Sunday, 2=Monday, …, 7=Saturday
DOW_SPARK = {
    "Lunes": 2, "Martes": 3, "Miércoles": 4, "Jueves": 5,
    "Viernes": 6, "Sábado": 7, "Domingo": 1,
}

FLEET_COLOR = {"Yellow": "#f7c430", "Green": "#2a9d5c"}
FLEET_FILL  = {"Yellow": "rgba(247,196,48,0.15)", "Green": "rgba(42,157,92,0.12)"}

st.set_page_config(
    page_title="NYC Taxi · Dispatch",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600&display=swap');

:root {
  --yellow:   #f7c430;
  --yellow-d: #d4a017;
  --navy:     #0f1729;
  --navy-mid: #1e2d4a;
  --black:    #111111;
  --white:    #ffffff;
  --bg:       #f5f4f0;
  --surf:     #ffffff;
  --bord:     #dddbd4;
  --bord-d:   #ccc9c0;
  --text:     #1a1a1a;
  --dim:      #888880;
  --green:    #2a9d5c;
  --red:      #e63946;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }
#MainMenu, header, footer { visibility: hidden; }

.block-container {
  padding-top: 0 !important;
  padding-bottom: 2rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--navy) !important;
  border-right: 3px solid var(--yellow) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
  color: rgba(255,255,255,0.7) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: rgba(255,255,255,0.7) !important;
}

/* ── Sidebar buttons ── */
[data-testid="stSidebar"] button {
  background: transparent !important;
  border: 1px solid rgba(247,196,48,0.5) !important;
  color: var(--yellow) !important;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.08em !important;
}
[data-testid="stSidebar"] button:hover {
  background: rgba(247,196,48,0.12) !important;
  border-color: var(--yellow) !important;
}
[data-testid="stSidebar"] button p {
  color: var(--yellow) !important;
}

/* ── Sidebar brand ── */
.sb-brand {
  padding: 20px 0 16px;
  border-bottom: 1px solid rgba(247,196,48,0.2);
  margin-bottom: 14px;
}
.sb-logo {
  font-family: 'Bebas Neue', sans-serif !important;
  font-size: 2rem;
  color: var(--yellow);
  letter-spacing: 0.06em;
  line-height: 1;
}
.sb-sub {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.52rem;
  color: rgba(255,255,255,0.35);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin-top: 4px;
}
.sb-sec {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.54rem;
  color: var(--yellow);
  text-transform: uppercase;
  letter-spacing: 0.16em;
  margin: 16px 0 6px;
  opacity: 0.85;
}

/* ── Status badge ── */
@keyframes dot-blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.badge-ok {
  background: rgba(42,157,92,0.15);
  border: 1px solid rgba(42,157,92,0.5);
  color: #4ade80;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.54rem;
  letter-spacing: 0.1em;
  padding: 3px 10px;
  display: inline-block;
}
.badge-ok .dot { animation: dot-blink 1.6s ease-in-out infinite; }
.badge-err {
  background: rgba(230,57,70,0.15);
  border: 1px solid rgba(230,57,70,0.5);
  color: #f87171;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.54rem;
  letter-spacing: 0.1em;
  padding: 3px 10px;
  display: inline-block;
}
.badge-err .dot { animation: dot-blink 0.7s ease-in-out infinite; }

/* ── App header ── */
.rw-header {
  background: var(--navy);
  padding: 28px 32px 22px;
  margin-bottom: 20px;
  display: flex;
  align-items: flex-end;
  gap: 24px;
  border-bottom: 4px solid var(--yellow);
}
.rw-header-left { flex: 1; }
.rw-title {
  font-family: 'Bebas Neue', sans-serif !important;
  font-size: 3.0rem;
  color: var(--yellow);
  letter-spacing: 0.05em;
  line-height: 0.92;
}
.rw-sub {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.56rem;
  color: rgba(255,255,255,0.38);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-top: 7px;
}
.rw-header-right {
  text-align: right;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.62rem;
  color: rgba(255,255,255,0.55);
  line-height: 1.9;
  flex-shrink: 0;
}
.rw-header-right strong {
  color: var(--yellow);
  font-weight: 600;
}

/* ── Route bar ── */
.route-bar {
  background: var(--white);
  border: 1px solid var(--bord);
  border-top: 3px solid var(--yellow);
  padding: 12px 20px;
  display: flex;
  align-items: center;
  gap: 10px;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.82rem;
  flex-wrap: wrap;
  margin-bottom: 16px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.r-fleet  { color: var(--dim); font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.1em; }
.r-origin { color: var(--navy); font-weight: 600; }
.r-dest   { color: var(--navy); font-weight: 600; }
.r-boro   { color: var(--dim); font-size: 0.65rem; }
.r-arrow  { color: var(--yellow-d); font-size: 1rem; }
.r-badge  {
  margin-left: auto;
  background: var(--bg);
  border: 1px solid var(--bord);
  padding: 3px 11px;
  font-size: 0.58rem;
  color: var(--dim);
}

/* ── KPI grid ── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
  margin-bottom: 8px;
}

@keyframes kpi-reveal {
  from { opacity: 0; transform: translateY(8px); }
  to   { opacity: 1; transform: none; }
}

.kpi-card {
  background: var(--white);
  border: 1px solid var(--bord);
  border-left: 4px solid var(--yellow);
  padding: 18px 16px 13px;
  animation: kpi-reveal 0.4s ease backwards;
  transition: box-shadow 0.2s, transform 0.2s, border-left-color 0.2s;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  cursor: default;
}
.kpi-card:hover {
  box-shadow: 0 6px 24px rgba(0,0,0,0.1);
  transform: translateY(-2px);
  border-left-color: var(--navy);
}
.kpi-tag {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.52rem;
  color: var(--dim);
  text-transform: uppercase;
  letter-spacing: 0.15em;
  margin-bottom: 8px;
}
.kpi-val {
  font-family: 'Bebas Neue', sans-serif !important;
  font-size: 2.7rem;
  color: var(--navy);
  line-height: 1;
  letter-spacing: 0.02em;
}
.kpi-unit {
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.6rem;
  color: var(--dim);
  margin-top: 1px;
}
.kpi-dpos { font-size: 0.58rem; color: var(--green); margin-top: 5px; font-weight: 500; }
.kpi-dneg { font-size: 0.58rem; color: var(--red);   margin-top: 5px; font-weight: 500; }
.kpi-dneu { font-size: 0.58rem; color: var(--dim);   margin-top: 5px; }

/* ── Section label ── */
.sec-lbl {
  font-family: 'Bebas Neue', sans-serif !important;
  font-size: 1.05rem;
  color: var(--navy);
  letter-spacing: 0.06em;
  margin: 1.3rem 0 0.4rem;
  display: flex;
  align-items: center;
  gap: 12px;
}
.sec-lbl::after {
  content: '';
  display: inline-block;
  height: 3px;
  width: 40px;
  background: var(--yellow);
  flex-shrink: 0;
}
.sec-lbl-line {
  font-family: 'Bebas Neue', sans-serif !important;
  font-size: 1.05rem;
  letter-spacing: 0.06em;
  margin: 0.7rem 0 0.3rem;
  display: flex;
  align-items: center;
  gap: 12px;
}
.sec-lbl-line::after {
  content: '';
  display: inline-block;
  height: 3px;
  width: 40px;
  flex-shrink: 0;
}
.sec-lbl-grn { color: var(--green); }
.sec-lbl-grn::after { background: var(--green); }
.sec-lbl-mag { color: var(--red); }
.sec-lbl-mag::after { background: var(--red); }

/* ── Model stats ── */
.model-row {
  background: var(--bg);
  border: 1px solid var(--bord);
  padding: 5px 10px;
  font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.58rem;
  color: var(--dim);
  display: flex;
  justify-content: space-between;
  margin-bottom: 3px;
}
.model-row b { color: var(--navy); font-weight: 600; }
.model-row.live b { color: var(--green); }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource
def load_spark():
    from pyspark.sql import SparkSession
    mlflow.set_tracking_uri(MLFLOW_URI)
    return SparkSession.builder.appName("StreamlitTaxiApp").getOrCreate()


@st.cache_resource
def load_models():
    mlflow.set_tracking_uri(MLFLOW_URI)
    return (
        mlflow.spark.load_model(f"models:/{MODEL_FARE}/Production"),
        mlflow.spark.load_model(f"models:/{MODEL_DURATION}/Production"),
    )


@st.cache_data
def load_zones():
    return pd.read_csv(ZONES_PATH)


@st.cache_data
def load_dist_media():
    return load_spark().read.parquet(DIST_MEDIA_PATH).toPandas()


@st.cache_data(ttl=3600)
def load_model_metrics() -> dict | None:
    """Pull live R²/MAE from the latest MLflow training run."""
    try:
        client = mlflow.MlflowClient(MLFLOW_URI)
        exp = client.get_experiment_by_name(EXPERIMENT)
        if not exp:
            return None
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.run_name = 'taxi_model_training'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None
        parent = runs[0]
        metrics: dict = {"total_rows": int(parent.data.metrics.get("total_rows", 0))}
        child_runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'",
        )
        for cr in child_runs:
            metrics.update(cr.data.metrics)
        return metrics
    except Exception:
        return None


# ── Prediction helpers ─────────────────────────────────────────────────────────
def _row(pu_id: int, do_id: int, dist: float, hour: int,
         dow: int, month: int, is_weekend: int, fleet_int: int) -> dict:
    return dict(
        PULocationID=int(pu_id), DOLocationID=int(do_id),
        trip_distance=float(dist), hour=int(hour), day_of_week=int(dow),
        month_num=int(month), is_weekend=int(is_weekend), fleet_int=int(fleet_int),
    )


def _sorted_preds(sdf, model, alias: str) -> list:
    return (model.transform(sdf)
            .select("_i", F.col("prediction").alias(alias))
            .toPandas().sort_values("_i")[alias].tolist())


def predict_batch(spark, m_fare, m_dur, rows: list) -> list:
    sdf   = spark.createDataFrame(rows).withColumn("_i", F.monotonically_increasing_id())
    fares = _sorted_preds(sdf, m_fare, "f")
    durs  = _sorted_preds(sdf, m_dur,  "d")
    return [(max(f, 3.0), max(d, 1.0)) for f, d in zip(fares, durs)]


def predict_fare_batch(spark, m_fare, rows: list) -> list:
    sdf = spark.createDataFrame(rows).withColumn("_i", F.monotonically_increasing_id())
    return [max(v, 3.0) for v in _sorted_preds(sdf, m_fare, "f")]


# ── Charts ─────────────────────────────────────────────────────────────────────
_BG   = dict(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
_AXIS = dict(gridcolor="#f0ede8", color="#888880", zerolinecolor="#e8e5e0")


def hourly_chart(preds_24: list, hora: int, fleet_name: str):
    fc   = FLEET_COLOR[fleet_name]
    fill = FLEET_FILL[fleet_name]
    hours = list(range(24))
    fares = [f for f, _ in preds_24]
    durs  = [d for _, d in preds_24]
    avg_f = np.mean(fares)

    fig = go.Figure()
    fig.add_hline(y=avg_f, line_dash="dot", line_color="rgba(0,0,0,0.15)",
                  annotation_text=f"  avg ${avg_f:.2f}",
                  annotation_position="top left",
                  annotation_font=dict(size=9, color="#888880",
                                       family="IBM Plex Mono"))
    fig.add_trace(go.Scatter(
        x=hours, y=fares, name="Fare ($)",
        mode="lines", fill="tozeroy",
        line=dict(color=fc, width=2.5), fillcolor=fill, yaxis="y1",
        hovertemplate="<b>%{x:02d}:00h</b><br>$%{y:.2f}<extra>Fare</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=hours, y=durs, name="Duration (min)",
        mode="lines", line=dict(color="#0f1729", width=1.8, dash="dot"), yaxis="y2",
        hovertemplate="<b>%{x:02d}:00h</b><br>%{y:.1f} min<extra>Duration</extra>",
    ))
    fig.add_vline(x=hora, line_dash="dash", line_color="rgba(15,23,41,0.25)", line_width=1.5)
    fig.add_trace(go.Scatter(
        x=[hora], y=[fares[hora]], mode="markers",
        marker=dict(color=fc, size=10, line=dict(color="#0f1729", width=2)),
        showlegend=False, yaxis="y1", hoverinfo="skip",
    ))
    fig.update_layout(
        **_BG, height=275, margin=dict(l=0, r=0, t=22, b=0),
        legend=dict(orientation="h", x=0, y=1.18, bgcolor="#ffffff",
                    font=dict(color="#1a1a1a", size=10, family="IBM Plex Mono")),
        xaxis=dict(**_AXIS, tickmode="linear", dtick=3, ticksuffix="h",
                   tickfont=dict(family="IBM Plex Mono", size=10)),
        yaxis=dict(**_AXIS, title="$", tickfont=dict(family="IBM Plex Mono", size=10)),
        yaxis2=dict(**_AXIS, title="min", overlaying="y", side="right",
                    tickfont=dict(family="IBM Plex Mono", size=10)),
        hovermode="x unified",
        font=dict(family="IBM Plex Mono"),
    )
    return fig


def dow_chart(preds_7: list, selected_dow: int, fleet_name: str):
    fc    = FLEET_COLOR[fleet_name]
    fares = [f for f, _ in preds_7]
    bar_c  = [fc if i == selected_dow else "#e8e5e0" for i in range(7)]
    text_c = ["#0f1729" if i == selected_dow else "#888880" for i in range(7)]

    fig = go.Figure(go.Bar(
        x=DAYS_SHORT, y=fares,
        marker_color=bar_c, marker_line_width=0,
        text=[f"${v:.0f}" for v in fares],
        textposition="outside",
        textfont=dict(color=text_c, size=10, family="IBM Plex Mono"),
        hovertemplate="%{x}<br>$%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_BG, height=275, margin=dict(l=0, r=0, t=22, b=0),
        xaxis=dict(color="#1a1a1a", gridcolor="rgba(0,0,0,0)",
                   tickfont=dict(size=11, family="DM Sans")),
        yaxis=dict(**_AXIS, title="$", tickfont=dict(family="IBM Plex Mono", size=10)),
        showlegend=False, bargap=0.38,
        font=dict(family="IBM Plex Mono"),
    )
    return fig


def top5_chart(df5: pd.DataFrame, color: str):
    df5 = df5.copy().reset_index(drop=True)
    max_fare = df5["Tarifa ($)"].max()
    fig = go.Figure(go.Bar(
        x=df5["Tarifa ($)"], y=df5["Zona"].str[:28],
        orientation="h", marker_color=color, marker_line_width=0,
        text=df5["Tarifa ($)"].apply(lambda v: f"${v:.2f}"),
        textposition="outside",
        textfont=dict(color="#1a1a1a", size=10, family="IBM Plex Mono"),
        hovertemplate="%{y}<br>$%{x:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_BG, height=200, margin=dict(l=0, r=55, t=0, b=0),
        xaxis=dict(**_AXIS, range=[0, max_fare * 1.40],
                   tickfont=dict(family="IBM Plex Mono", size=10)),
        yaxis=dict(color="#1a1a1a", tickfont=dict(size=10, family="DM Sans"),
                   autorange="reversed"),
        font=dict(family="IBM Plex Mono"),
    )
    return fig


# ── KPI helpers ────────────────────────────────────────────────────────────────
def _delta(val: float, ref: float, fmt: str, lower_better: bool = False) -> str:
    diff = val - ref
    if abs(diff) < 0.01 * max(abs(ref), 0.01):
        return '<div class="kpi-dneu">— vs daily avg</div>'
    pct  = diff / ref * 100
    sign = "+" if diff > 0 else ""
    good = (diff < 0) if lower_better else (diff > 0)
    cls  = "kpi-dpos" if good else "kpi-dneg"
    arrow = "↑" if diff > 0 else "↓"
    return f'<div class="{cls}">{arrow} {sign}{fmt.format(diff)} ({sign}{pct:.0f}%)</div>'


def _kpi(tag: str, value: str, unit: str, delta: str = "", delay: float = 0.0) -> str:
    return (
        f'<div class="kpi-card" style="animation-delay:{delay}s">'
        f'<div class="kpi-tag">{tag}</div>'
        f'<div class="kpi-val">{value}</div>'
        f'<div class="kpi-unit">{unit}</div>'
        f'{delta}</div>'
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    zones_df = load_zones()
    dist_df  = load_dist_media()
    spark    = load_spark()

    m_fare = m_dur = None
    with st.spinner("Loading models…"):
        try:
            m_fare, m_dur = load_models()
            model_ok = True
        except Exception as e:
            st.error(f"Model load failed: {e}")
            st.info("Run train.py and promote models to 'Production' in MLflow.")
            return

    zone_names = zones_df["Zone"].tolist()
    zone_ids   = dict(zip(zones_df["Zone"], zones_df["LocationID"]))

    default_pu = zone_names.index("JFK Airport") if "JFK Airport" in zone_names else 0
    default_do = (zone_names.index("Times Sq/Theatre District")
                  if "Times Sq/Theatre District" in zone_names else 1)

    if "pu_idx" not in st.session_state:
        st.session_state.pu_idx = default_pu
    if "do_idx" not in st.session_state:
        st.session_state.do_idx = default_do

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        badge = (
            '<span class="badge-ok"><span class="dot">●</span> ONLINE</span>'
            if model_ok else
            '<span class="badge-err"><span class="dot">●</span> OFFLINE</span>'
        )
        st.markdown(
            f'<div class="sb-brand">'
            f'<div class="sb-logo">NYC Taxi</div>'
            f'<div class="sb-sub">Dispatch Intelligence</div>'
            f'<div style="margin-top:10px;">{badge}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sb-sec">Route</div>', unsafe_allow_html=True)
        zona_origen  = st.selectbox("Origen",  zone_names,
                                    index=st.session_state.pu_idx,
                                    label_visibility="collapsed")
        zona_destino = st.selectbox("Destino", zone_names,
                                    index=st.session_state.do_idx,
                                    label_visibility="collapsed")

        st.session_state.pu_idx = zone_names.index(zona_origen)
        st.session_state.do_idx = zone_names.index(zona_destino)

        _, mid, _ = st.columns([1, 2, 1])
        with mid:
            if st.button("⇅ Swap", use_container_width=True,
                         help="Swap origin and destination"):
                st.session_state.pu_idx, st.session_state.do_idx = (
                    zone_names.index(zona_destino),
                    zone_names.index(zona_origen),
                )
                st.rerun()

        st.markdown('<div class="sb-sec">Time</div>', unsafe_allow_html=True)
        hora    = st.slider("Hour", 0, 23, 8, format="%dh")
        dia_sem = st.selectbox("Day", DAYS_FULL)
        mes     = st.selectbox("Month", list(range(1, 13)), index=0,
                               format_func=lambda x: MONTHS[x - 1])

        st.markdown('<div class="sb-sec">Fleet</div>', unsafe_allow_html=True)
        flota = st.radio("Fleet", ["Yellow", "Green"], horizontal=True,
                         label_visibility="collapsed")

        st.divider()
        with st.expander("Model Metrics"):
            mlx = load_model_metrics()
            if mlx:
                r2_fare  = mlx.get(f"{MODEL_FARE}_r2",  0)
                mae_fare = mlx.get(f"{MODEL_FARE}_mae", 0)
                r2_dur   = mlx.get(f"{MODEL_DURATION}_r2",  0)
                mae_dur  = mlx.get(f"{MODEL_DURATION}_mae", 0)
                n_rows   = mlx.get("total_rows", 0)
                st.markdown(
                    f'<div class="model-row live">R² fare      <b>{r2_fare:.3f}</b></div>'
                    f'<div class="model-row live">MAE fare     <b>${mae_fare:.2f}</b></div>'
                    f'<div class="model-row live">R² duration  <b>{r2_dur:.3f}</b></div>'
                    f'<div class="model-row live">MAE duration <b>{mae_dur:.2f} min</b></div>'
                    f'<div class="model-row live">Dataset      <b>{n_rows:,.0f} trips</b></div>'
                    f'<div class="model-row">Period       <b>NYC TLC 2024</b></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="model-row">R² fare      <b>0.938</b></div>'
                    '<div class="model-row">MAE fare     <b>$2.12</b></div>'
                    '<div class="model-row">R² duration  <b>0.767</b></div>'
                    '<div class="model-row">MAE duration <b>3.35 min</b></div>'
                    '<div class="model-row">Dataset      <b>2.7 M trips</b></div>'
                    '<div class="model-row">Period       <b>NYC TLC 2024</b></div>',
                    unsafe_allow_html=True,
                )

    # ── Derived inputs ────────────────────────────────────────────────────────
    pu_id     = zone_ids[zona_origen]
    do_id     = zone_ids[zona_destino]
    dow       = DOW_SPARK[dia_sem]
    is_we     = 1 if dia_sem in ("Sábado", "Domingo") else 0
    fleet_int = 1 if flota == "Yellow" else 0

    pair_dist = dist_df[(dist_df["PULocationID"] == pu_id) & (dist_df["DOLocationID"] == do_id)]
    trip_dist = float(pair_dist["dist_media_miles"].values[0]) if len(pair_dist) > 0 else 5.0
    dist_km   = trip_dist * 1.60934
    dist_est  = len(pair_dist) == 0

    boro_pu = zones_df.loc[zones_df["Zone"] == zona_origen,  "Borough"].values
    boro_do = zones_df.loc[zones_df["Zone"] == zona_destino, "Borough"].values
    boro_pu = boro_pu[0] if len(boro_pu) else "–"
    boro_do = boro_do[0] if len(boro_do) else "–"

    # ── App header ────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="rw-header">'
        f'<div class="rw-header-left">'
        f'<div class="rw-title">NYC Taxi Dispatch AI</div>'
        f'<div class="rw-sub">Apache Spark 3.5.1 · GBTRegressor · MLflow · NYC TLC 2024</div>'
        f'</div>'
        f'<div class="rw-header-right">'
        f'{dia_sem.upper()} / {MONTHS[mes-1].upper()}<br>'
        f'<strong>{hora:02d}:00</strong> · {flota.upper()} FLEET'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Route bar ─────────────────────────────────────────────────────────────
    est_label = ' <span style="font-size:0.54rem;color:var(--dim);">(est.)</span>' if dist_est else ""
    st.markdown(
        f'<div class="route-bar">'
        f'<span class="r-fleet">{"🟡" if flota == "Yellow" else "🟢"} {flota}</span>'
        f'<span style="color:var(--bord-d);">│</span>'
        f'<span class="r-origin">{zona_origen}</span>'
        f'<span class="r-boro"> / {boro_pu}</span>'
        f'<span class="r-arrow"> → </span>'
        f'<span class="r-dest">{zona_destino}</span>'
        f'<span class="r-boro"> / {boro_do}</span>'
        f'<span class="r-badge">{trip_dist:.1f} mi · {dist_km:.1f} km{est_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Batch: 24 hourly + 7 DOW predictions ─────────────────────────────────
    hourly_rows = [
        _row(pu_id, do_id, trip_dist, h, dow, mes, is_we, fleet_int)
        for h in range(24)
    ]
    dow_rows = [
        _row(pu_id, do_id, trip_dist, hora,
             DOW_SPARK[DAYS_FULL[d]], mes,
             1 if d >= 5 else 0, fleet_int)
        for d in range(7)
    ]

    with st.spinner("Running predictions…"):
        all_preds = predict_batch(spark, m_fare, m_dur, hourly_rows + dow_rows)

    preds_24 = all_preds[:24]
    preds_7  = all_preds[24:]

    fare, dur  = preds_24[hora]
    avg_fare   = np.mean([f for f, _ in preds_24])
    avg_dur    = np.mean([d for _, d in preds_24])
    usd_per_km = fare / dist_km if dist_km > 0 else 0.0
    speed_kmh  = dist_km / (dur / 60) if dur > 0 else 0.0

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown(
        '<div class="kpi-grid">'
        + _kpi("Estimated fare",     f"${fare:.2f}", "",
               _delta(fare, avg_fare, "${:+.2f}", lower_better=True), delay=0.0)
        + _kpi("Estimated duration", f"{dur:.1f}",   "min",
               _delta(dur,  avg_dur, "{:+.1f} min",  lower_better=True), delay=0.1)
        + _kpi("Cost per km",        f"${usd_per_km:.2f}", "/ km", delay=0.2)
        + _kpi("Avg speed",          f"{speed_kmh:.1f}",   "km/h", delay=0.3)
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    col_h, col_d = st.columns([3, 2])
    with col_h:
        st.markdown('<div class="sec-lbl">Fare by hour of day</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(hourly_chart(preds_24, hora, flota), use_container_width=True)
    with col_d:
        st.markdown('<div class="sec-lbl">Fare by day of week</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(dow_chart(preds_7, DAYS_FULL.index(dia_sem), flota),
                        use_container_width=True)

    # ── Top-5 destinations ────────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl">Top destinations from this origin</div>',
                unsafe_allow_html=True)

    boroughs    = ["All boroughs"] + sorted(zones_df["Borough"].dropna().unique().tolist())
    boro_filter = st.selectbox("Filter by borough", boroughs,
                               label_visibility="collapsed")

    zone_rows, zone_meta = [], []
    filtered_z = (zones_df if boro_filter == "All boroughs"
                  else zones_df[zones_df["Borough"] == boro_filter])

    for _, row in filtered_z.iterrows():
        if row["LocationID"] == pu_id:
            continue
        pd_row = dist_df[(dist_df["PULocationID"] == pu_id) &
                         (dist_df["DOLocationID"] == row["LocationID"])]
        d = float(pd_row["dist_media_miles"].values[0]) if len(pd_row) > 0 else 5.0
        zone_rows.append(_row(pu_id, int(row["LocationID"]), d, hora, dow, mes, is_we, fleet_int))
        zone_meta.append({"Zona": row["Zone"], "Borough": row["Borough"],
                          "dist_mi": round(d, 2)})

    with st.spinner("Scoring destinations…"):
        fare_preds = predict_fare_batch(spark, m_fare, zone_rows)

    preds_df = pd.DataFrame([
        {**m, "Tarifa ($)": round(f, 2)} for m, f in zip(zone_meta, fare_preds)
    ]).sort_values("Tarifa ($)").reset_index(drop=True)

    cheapest = preds_df.head(5)
    priciest = preds_df.tail(5).sort_values("Tarifa ($)", ascending=False)

    _col_cfg = {
        "Tarifa ($)": st.column_config.NumberColumn(format="$%.2f"),
        "Dist (mi)":  st.column_config.NumberColumn(format="%.2f mi"),
    }

    col_cheap, col_price = st.columns(2)

    with col_cheap:
        st.markdown(
            '<div class="sec-lbl-line sec-lbl-grn">Cheapest 5 destinations</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(top5_chart(cheapest, "#2a9d5c"), use_container_width=True)
        st.dataframe(
            cheapest[["Zona", "Borough", "dist_mi", "Tarifa ($)"]].rename(
                columns={"dist_mi": "Dist (mi)"}
            ).reset_index(drop=True),
            use_container_width=True, hide_index=True, column_config=_col_cfg,
        )

    with col_price:
        st.markdown(
            '<div class="sec-lbl-line sec-lbl-mag">Priciest 5 destinations</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(top5_chart(priciest, "#e63946"), use_container_width=True)
        st.dataframe(
            priciest[["Zona", "Borough", "dist_mi", "Tarifa ($)"]].rename(
                columns={"dist_mi": "Dist (mi)"}
            ).reset_index(drop=True),
            use_container_width=True, hide_index=True, column_config=_col_cfg,
        )


if __name__ == "__main__":
    main()
