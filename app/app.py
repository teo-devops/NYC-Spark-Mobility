"""
app/app.py
──────────
App Streamlit NYC Taxi AI — versión Distributed Lab
· Carga modelos directamente desde MLflow Model Registry (no pkl)
· Carga dist_media desde Gold/dist_media Parquet
· Mantiene toda la lógica visual de taxi-app-ai (mapa boroughs, tabla Top-5, etc.)
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import mlflow.spark
from pyspark.sql import SparkSession

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI        = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_FARE        = os.getenv("MODEL_NAME_FARE",     "xgb_fare")
MODEL_DURATION    = os.getenv("MODEL_NAME_DURATION", "xgb_duration")
DIST_MEDIA_PATH   = os.getenv("DIST_MEDIA_PATH",     "/app/data/gold/dist_media")
ZONES_PATH        = os.getenv("ZONES_PATH",           "/app/data/bronze/taxi_zone_lookup.csv")

FEATURE_COLS = [
    "PULocationID", "DOLocationID", "trip_distance",
    "hour", "day_of_week", "month_num", "is_weekend", "fleet_int",
]

# Mapa boroughs → coordenadas aproximadas (para mapa Plotly)
BOROUGH_COORDS = {
    "Manhattan":    (40.7831, -73.9712),
    "Brooklyn":     (40.6782, -73.9442),
    "Queens":       (40.7282, -73.7949),
    "Bronx":        (40.8448, -73.8648),
    "Staten Island":(40.5795, -74.1502),
    "EWR":          (40.6895, -74.1745),
}

st.set_page_config(page_title="NYC Taxi AI", page_icon="🚕", layout="wide")


# ── Cache recursos ────────────────────────────────────────────────────────────

@st.cache_resource
def load_spark():
    mlflow.set_tracking_uri(MLFLOW_URI)
    return SparkSession.builder.appName("StreamlitTaxiApp").getOrCreate()


@st.cache_resource
def load_models():
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_fare     = mlflow.spark.load_model(f"models:/{MODEL_FARE}/Production")
    model_duration = mlflow.spark.load_model(f"models:/{MODEL_DURATION}/Production")
    return model_fare, model_duration


@st.cache_data
def load_zones():
    return pd.read_csv(ZONES_PATH)


@st.cache_data
def load_dist_media():
    spark = load_spark()
    return spark.read.parquet(DIST_MEDIA_PATH).toPandas()


# ── Predicción ─────────────────────────────────────────────────────────────────

def predict(spark, model_fare, model_duration, pu, do_, trip_dist, hour, dow, month, is_weekend, fleet_int):
    import pandas as pd
    from pyspark.ml.feature import VectorAssembler

    row = pd.DataFrame([{
        "PULocationID": pu,
        "DOLocationID": do_,
        "trip_distance": trip_dist,
        "hour": hour,
        "day_of_week": dow,
        "month_num": month,
        "is_weekend": is_weekend,
        "fleet_int": fleet_int,
    }])
    sdf = spark.createDataFrame(row)
    fare     = model_fare.transform(sdf).select("prediction").collect()[0][0]
    duration = model_duration.transform(sdf).select("prediction").collect()[0][0]
    return max(fare, 3.0), max(duration, 1.0)


# ── UI ─────────────────────────────────────────────────────────────────────────

def main():
    st.title("🚕 NYC Taxi AI — Distributed Lab")
    st.caption("Modelos entrenados con Spark ML sobre 7M+ viajes · MLflow Model Registry")

    zones_df    = load_zones()
    dist_df     = load_dist_media()
    spark       = load_spark()

    with st.spinner("Cargando modelos desde MLflow..."):
        try:
            model_fare, model_duration = load_models()
            st.success("✅ Modelos cargados desde MLflow Model Registry")
        except Exception as e:
            st.error(f"❌ Error cargando modelos: {e}")
            st.info("Asegúrate de haber ejecutado train.py y de que los modelos estén en 'Production'.")
            return

    zone_names = zones_df["Zone"].tolist()
    zone_ids   = dict(zip(zones_df["Zone"], zones_df["LocationID"]))

    # ── Sidebar de inputs ─────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🗺️ Configurar viaje")
        zona_origen  = st.selectbox("Origen",  zone_names, index=zone_names.index("JFK Airport") if "JFK Airport" in zone_names else 0)
        zona_destino = st.selectbox("Destino", zone_names, index=zone_names.index("Times Sq/Theatre District") if "Times Sq/Theatre District" in zone_names else 1)
        hora    = st.slider("Hora del día", 0, 23, 8)
        dia_sem = st.selectbox("Día de la semana", ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"], index=0)
        mes     = st.selectbox("Mes", list(range(1, 13)), index=2, format_func=lambda x: f"{x:02d}")
        flota   = st.radio("Flota", ["Yellow", "Green"])

    pu_id  = zone_ids[zona_origen]
    do_id  = zone_ids[zona_destino]
    dow    = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"].index(dia_sem) + 2  # Spark: 1=Dom
    is_we  = 1 if dia_sem in ["Sábado", "Domingo"] else 0
    fleet_int = 1 if flota == "Yellow" else 0

    # Distancia automática desde Gold
    pair_dist = dist_df[(dist_df["PULocationID"] == pu_id) & (dist_df["DOLocationID"] == do_id)]
    trip_dist = float(pair_dist["dist_media_miles"].values[0]) if len(pair_dist) > 0 else 5.0

    st.markdown(f"**Distancia estimada:** {trip_dist:.2f} millas *(media histórica del par PU/DO)*")

    # ── Predicción principal ──────────────────────────────────────────────────
    fare, duration = predict(spark, model_fare, model_duration,
                             pu_id, do_id, trip_dist, hora, dow, mes, is_we, fleet_int)

    dist_km      = trip_dist * 1.60934
    usd_per_km   = fare / dist_km if dist_km > 0 else 0
    min_per_km   = duration / dist_km if dist_km > 0 else 0
    speed_kmh    = dist_km / (duration / 60) if duration > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Tarifa estimada",    f"${fare:.2f}")
    col2.metric("⏱️ Duración estimada", f"{duration:.1f} min")
    col3.metric("💹 $/km",              f"${usd_per_km:.2f}")
    col4.metric("🚀 Velocidad media",   f"{speed_kmh:.1f} km/h")

    # ── Gráfico por hora ──────────────────────────────────────────────────────
    st.subheader("📈 Estimación por hora del día para esta ruta")
    hourly_data = []
    for h in range(24):
        f, d = predict(spark, model_fare, model_duration,
                       pu_id, do_id, trip_dist, h, dow, mes, is_we, fleet_int)
        hourly_data.append({"Hora": h, "Tarifa ($)": f, "Duración (min)": d})

    hourly_df = pd.DataFrame(hourly_data)
    fig = px.line(hourly_df, x="Hora", y=["Tarifa ($)", "Duración (min)"],
                  title=f"{zona_origen} → {zona_destino}",
                  color_discrete_sequence=["#E25A1C", "#1C7AE2"])
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    # ── Top 5 destinos ────────────────────────────────────────────────────────
    st.subheader("🏆 Top 5 destinos más baratos y más caros desde este origen")
    preds = []
    for _, row in zones_df.iterrows():
        if row["LocationID"] == pu_id:
            continue
        pd_row = dist_df[(dist_df["PULocationID"] == pu_id) & (dist_df["DOLocationID"] == row["LocationID"])]
        d = float(pd_row["dist_media_miles"].values[0]) if len(pd_row) > 0 else 5.0
        f, _ = predict(spark, model_fare, model_duration,
                       pu_id, int(row["LocationID"]), d, hora, dow, mes, is_we, fleet_int)
        preds.append({"Zona": row["Zone"], "Borough": row["Borough"], "Tarifa ($)": round(f, 2)})

    preds_df = pd.DataFrame(preds).sort_values("Tarifa ($)")
    col_a, col_b = st.columns(2)
    col_a.markdown("**5 más baratos**")
    col_a.dataframe(preds_df.head(5).reset_index(drop=True), use_container_width=True)
    col_b.markdown("**5 más caros**")
    col_b.dataframe(preds_df.tail(5).reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()
