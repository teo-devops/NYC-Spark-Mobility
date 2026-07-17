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
    "maxIter":    50,    # 300 causes StackOverflowError during task serialization
    "maxDepth":   5,
    "subsamplingRate": 0.8,
    "stepSize":   0.1,
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
        # dfs_tmpdir must be on the shared volume so worker-written parquet
        # files are accessible from the driver when MLflow copies artifacts.
        mlflow.spark.log_model(
            model,
            artifact_path=model_name,
            registered_model_name=model_name,
            dfs_tmpdir="/mlflow/tmp_spark_models",
        )

        print(f"   ✅ Modelo '{model_name}' registrado en MLflow")

        # Promover automáticamente a la etapa "Production"
        try:
            client = mlflow.tracking.MlflowClient(MLFLOW_URI)
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                latest_version = latest_versions[0].version
                print(f"   🚀 Promocionando versión {latest_version} de '{model_name}' a etapa 'Production'...")
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                print(f"   ✅ Modelo '{model_name}' promocionado con éxito a 'Production'")
        except Exception as e:
            print(f"   ⚠️  No se pudo promocionar el modelo '{model_name}' a Production: {e}")

        return model, metrics


def main():
    spark = get_spark("Train_Medallion_Models", extra_config={
        # GBT serialization produces deeply nested objects — increase JVM stack
        "spark.driver.extraJavaOptions":   "-Xss8m",
        "spark.executor.extraJavaOptions": "-Xss8m",
    })
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
