FROM apache/airflow:3.2.2

ENV TZ=Asia/Tehran

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential g++ gcc python3-dev curl \
        libffi-dev libkrb5-dev krb5-multidev krb5-user pkg-config \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 --retries=10 --user \
        "pandas>=2.2.3" \
        "numpy>=1.26.0" \
        "pyarrow>=17.0.0" \
        "duckdb>=1.1.0" \
        "polars>=1.8.0" \
        "pydantic>=2.8.0" \
        "s3fs>=2024.6.0" \
        "mlflow>=2.13.0" \
        "scikit-learn>=1.4.0" \
        "joblib>=1.3.0" \
        "redis>=5.0.0" \
        "psycopg2-binary>=2.9.9" \
        "requests>=2.32.0" \
        "pendulum>=3.0.0"
