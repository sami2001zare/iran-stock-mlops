FROM apache/airflow:3.2.2

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        g++ \
        gcc \
        python3-dev \
        libffi-dev \
        libkrb5-dev \
        krb5-multidev \
        krb5-user \
        pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Verify krb5-config exists (should be in /usr/bin)
RUN ls -la /usr/bin/krb5-config

ENV TZ=Asia/Tehran
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

USER airflow
RUN pip install --no-cache-dir --user \
        jdatetime==4.1.0 \
        openpyxl==3.1.2 \
        pandas>=2.2.3 \
        azure-storage-blob==12.19.0 \
        apache-airflow-providers-apache-hdfs==4.1.0