FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Tehran \
    PYTHONPATH=/app:/opt/airflow:/model
RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=1000 --retries=10 \
        "pandas>=2.2.3" "numpy>=1.26.0" "scikit-learn>=1.4.0" "joblib>=1.3.0" \
        "pyarrow>=17.0.0" "duckdb>=1.1.0" "polars>=1.8.0" "pydantic>=2.8.0" \
        "s3fs>=2024.6.0" "redis>=5.0.0" "psycopg2-binary>=2.9.9"
WORKDIR /app

FROM base AS mlflow
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 "mlflow>=2.13.0"

FROM base AS fastapi
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
        "mlflow>=2.13.0" \
        "fastapi>=0.111.0" "uvicorn[standard]>=0.29.0" "httpx>=0.27.0" \
        "prometheus-fastapi-instrumentator>=7.0.0"

FROM base AS tests
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
        "mlflow>=2.13.0" \
        "fastapi>=0.111.0" "uvicorn[standard]>=0.29.0" "httpx>=0.27.0" \
        "prometheus-fastapi-instrumentator>=7.0.0" \
        pytest pytest-asyncio pytest-cov ruff mypy
