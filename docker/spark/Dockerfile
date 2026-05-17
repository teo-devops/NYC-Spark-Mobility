FROM apache/spark:3.5.1

USER root

# Python packages required by ETL jobs and model training
RUN pip3 install --no-cache-dir \
    mlflow==2.11.3 \
    pymysql==1.1.0
