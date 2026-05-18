# docker/jupyter/Dockerfile
# ─────────────────────────────────────────────
# Base: quay.io/jupyter/pyspark-notebook (JupyterLab + PySpark + Java pre-baked)
# Same approach as spark_ML project (pytorch-notebook base).
# Only adds project-specific packages on top.
# ─────────────────────────────────────────────

FROM quay.io/jupyter/pyspark-notebook:spark-3.5.0

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl docker.io \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Project-specific packages not in the base image
RUN pip install --no-cache-dir \
    mlflow==2.11.3 \
    pymysql==1.1.0 \
    xgboost==2.0.3 \
    plotly==5.22.0 \
    seaborn==0.13.2 \
    streamlit==1.35.0

RUN chown -R jovyan:users /home/jovyan/work

USER jovyan
WORKDIR /home/jovyan/work

EXPOSE 8888
