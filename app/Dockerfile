FROM nyc-spark-mobility-spark

USER root

ENV PYSPARK_PYTHON=python3
ENV JAVA_HOME=/opt/java/openjdk
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
# mlflow, pymysql, pyspark already in base image — pip skips them
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8501

# curl not available in apache/spark base — use python for healthcheck
HEALTHCHECK CMD python3 -c \
  "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" \
  || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
