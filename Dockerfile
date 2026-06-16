FROM apache/airflow:3.2.2

ENV TZ=Asia/Tehran

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ gcc python3-dev \
        libffi-dev libkrb5-dev krb5-multidev krb5-user pkg-config \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir --user \
        jdatetime==4.1.0 \
        openpyxl==3.1.2 \
        "pandas>=2.2.3" \
        azure-storage-blob==12.19.0 \
        apache-airflow-providers-apache-hdfs==4.1.0 \
        "mlflow>=2.13.0" \
        "scikit-learn>=1.4.0" \
        "numpy>=1.26.0" \
        "joblib>=1.3.0"
