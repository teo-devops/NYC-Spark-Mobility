"""
src/models/train.py
────────────────────
Entrenamiento distribuido con Spark ML Pipeline.
· Lee Gold/features (tabla ABT)
· Entrena dos modelos independientes: tarifa y duración
· Pipeline: VectorAssembler → StandardScaler → GBTRegressor (o XGBoostEstimator)
· Registra params, métricas y modelo en MLflow Model Registry
· Replica los hiperparámetros del Notebook 02_predictive.ipynb de taxi-app-ai

Para usar XGBoost nativo de Spark (sparkxgb) descomentar las líneas indicadas.
Por defecto usa GBTRegressor de Spark ML (sin dependencia extra).
"""

import os
import sys

sys.path.insert(0, "/opt/spark/src")

import mlflow
import mlflow.spark
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

from common.utils import get_spark

# ── Config ─────────────────────────────────────────────────────────────────────
GOLD_FEATURES  = "/opt/spark/data/gold/features"
MLFLOW_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT     = "nyc_taxi_model"
TRAIN_RATIO    = 0.8
SEED           = 42

# Hiperparámetros — espejo de los usados en XGBoost del notebook
# GBTRegressor es el equivalente nativo de Spark
GBT_PARAMS = {
    "maxIter":    300,   # n_estimators
    "maxDepth":   6,
    "subsamplingRate": 0.8,
    "stepSize":   0.1,   # learning_rate
}

FEATURE_COLS = [
    "PULocationID", "DOLocationID", "trip_distance",
    "hour", "day_of_week", "month_num", "is_weekend", "fleet_int",
]


def build_pipeline(label_col: str) -> Pipeline:
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="raw_features")
    scaler    = StandardScaler(inputCol="raw_features", outputCol="features",
                               withMean=True, withStd=True)
    gbt = GBTRegressor(
        featuresCol="features",
        labelCol=label_col,
        predictionCol="prediction",
        **GBT_PARAMS,
    )
    return Pipeline(stages=[assembler, scaler, gbt])


def evaluate(predictions, label_col: str) -> dict:
    evaluator = RegressionEvaluator(labelCol=label_col, predictionCol="prediction")
    metrics = {}
    for metric in ["rmse", "mae", "r2"]:
        evaluator.setMetricName(metric)
        metrics[metric] = evaluator.evaluate(predictions)
    return metrics


def train_model(spark, df, label_col: str, model_name: str):
    """Entrena, evalúa y registra un modelo en MLflow."""
    train_df, test_df = df.randomSplit([TRAIN_RATIO, 1 - TRAIN_RATIO], seed=SEED)
    train_df.cache()
    test_df.cache()

    pipeline = build_pipeline(label_col)

    with mlflow.start_run(run_name=f"train_{model_name}", nested=True) as run:
        mlflow.log_params({
            "label":          label_col,
            "features":       FEATURE_COLS,
            "train_rows":     train_df.count(),
            "test_rows":      test_df.count(),
            **{f"gbt_{k}": v for k, v in GBT_PARAMS.items()},
        })

        print(f"🏋️  Entrenando modelo: {model_name}  (label={label_col})")
        model = pipeline.fit(train_df)

        predictions = model.transform(test_df)
        metrics = evaluate(predictions, label_col)

        mlflow.log_metrics({
            f"{model_name}_rmse": round(metrics["rmse"], 4),
            f"{model_name}_mae":  round(metrics["mae"],  4),
            f"{model_name}_r2":   round(metrics["r2"],   4),
        })

        print(f"   RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.4f}")

        # Registrar en MLflow Model Registry
        mlflow.spark.log_model(
            model,
            artifact_path=model_name,
            registered_model_name=model_name,
        )

        print(f"   ✅ Modelo '{model_name}' registrado en MLflow")
        return model, metrics


def main():
    spark = get_spark("Train_Medallion_Models")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    gold = spark.read.parquet(GOLD_FEATURES)
    gold.cache()
    print(f"📦 Gold features cargado: {gold.count():,} filas")

    with mlflow.start_run(run_name="taxi_model_training") as parent_run:
        mlflow.log_param("gold_path", GOLD_FEATURES)
        mlflow.log_metric("total_rows", gold.count())

        # Modelo 1: tarifa
        model_fare, metrics_fare = train_model(
            spark, gold,
            label_col="fare_amount",
            model_name="xgb_fare",
        )

        # Modelo 2: duración
        model_duration, metrics_duration = train_model(
            spark, gold,
            label_col="duration_min",
            model_name="xgb_duration",
        )

        # Resumen
        print("\n📊 Resumen final:")
        print(f"   Tarifa    — R²={metrics_fare['r2']:.3f}  MAE={metrics_fare['mae']:.2f} $")
        print(f"   Duración  — R²={metrics_duration['r2']:.3f}  MAE={metrics_duration['mae']:.2f} min")

    spark.stop()


if __name__ == "__main__":
    main()
