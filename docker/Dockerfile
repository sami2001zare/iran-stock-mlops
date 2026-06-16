FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Tehran
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
        "pandas>=2.2.3" "numpy>=1.26.0" "scikit-learn>=1.4.0" "joblib>=1.3.0"
WORKDIR /app

FROM base AS mlflow
RUN pip install --no-cache-dir "mlflow>=2.13.0" psycopg2-binary

FROM base AS fastapi
RUN pip install --no-cache-dir \
        "mlflow>=2.13.0" \
        "fastapi>=0.111.0" "uvicorn[standard]>=0.29.0" "pydantic>=2.7.0" "httpx>=0.27.0"

FROM base AS tests
RUN pip install --no-cache-dir \
        "mlflow>=2.13.0" \
        "fastapi>=0.111.0" "uvicorn[standard]>=0.29.0" "pydantic>=2.7.0" "httpx>=0.27.0" \
        pytest pytest-asyncio
