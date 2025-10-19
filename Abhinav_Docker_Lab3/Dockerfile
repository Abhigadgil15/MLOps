FROM python:3.11

ENV MLFLOW_HOME=/root/.mlflow

# Install system dependencies
RUN apt-get update && apt-get install -y curl && \
    pip install --no-cache-dir --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql://mlflow:mlflow@postgres:5432/mlflowdb", \
     "--default-artifact-root", "file:///root/.mlflow/artifacts"]
