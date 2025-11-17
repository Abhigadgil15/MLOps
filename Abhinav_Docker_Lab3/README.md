# Docker Lab using MLflow

This lab demonstrates running an **MLflow tracking server** with a **PostgreSQL backend** using Docker and Docker Compose. You will learn how to log experiments, track metrics, and view results in the MLflow UI, all in a **self-contained Docker environment**.

---

## 1️⃣ What is MLflow?

**MLflow** is an open-source platform designed to manage the **machine learning lifecycle**, including:

- **Experiment Tracking:** Log parameters, metrics, and artifacts from ML experiments.
- **Model Management:** Register, version, and stage ML models (e.g., staging → production).
- **Reproducibility:** Keep a record of code, data, parameters, and results.
- **Artifact Storage:** Store output files like trained models, plots, or datasets.

Think of MLflow as a **"GitHub for ML experiments"** — it tracks everything you do in training so you can compare experiments, reproduce results, and deploy models safely.

---

## 2️⃣ Lab Overview

This lab sets up:

- **MLflow Server:** A web-based tracking server that hosts the MLflow UI, listening on a configurable port (default `5000`).
- **PostgreSQL:** Stores MLflow metadata (experiments, runs, parameters, metrics).
- **Dockerized Environment:** Ensures the setup is isolated, reproducible, and portable.

**Note:** Artifact storage is local (`/root/.mlflow`) for simplicity — no cloud storage required.

---

## 3️⃣ Prerequisites

- Docker ≥ 20.x
- Docker Compose ≥ 1.29.x
- Optional: Python ≥ 3.8 for running experiment scripts from your host

---

## 4️⃣ Directory Structure
```
Abhinav_Docker_Lab3/
│
├── Dockerfile              # MLflow server Dockerfile
├── requirements.txt        # Python dependencies
├── ml_flow.py              # This file prints the metrics for a model
├── docker-compose.yml      # Compose file to start MLflow + Postgres
└── README.md               # This documentation file
```

---

## 5️⃣ Setup and Run

### 5.1 Clone the repository
```bash
git clone https://github.com/Abhigadgil15/MLOps.git
cd Abhinav_Docker_Lab3
```

### 5.2 Build and start MLflow server
```bash
docker-compose up --build
```

**What happens:**

- `docker-compose up` starts all services defined in `docker-compose.yml`:
  - `mlflow-server` (MLflow UI)
  - `postgres` (PostgreSQL backend)
- `--build` ensures that any changes in the Dockerfile or dependencies are reflected in the image.

### 5.3 Access the MLflow UI

Open your browser at **http://localhost:5000**

You can view experiments, runs, parameters, metrics, and artifacts.

### 5.4 Stopping the environment
```bash
docker-compose down
```

**What happens:**

- Stops all running containers.
- Removes networks and temporary containers.

**Important:** Docker images are not deleted — rebuilding is faster next time.

**Note:** You do not need to run `python ml_flow.py` separately; MLflow experiments are logged by the scripts or notebooks you execute inside the container, and the MLflow server automatically tracks them.

### 5.5 Running experiments locally (optional)

If you want to test ML scripts locally:
```bash
python3 ml_flow.py
```

This logs metrics and parameters to the MLflow server (make sure it's running).

You can then refresh the UI to see updated experiment data.

---

## 6️⃣ Notes

### Port conflicts

If port 5000 is already in use, change the port mapping in `docker-compose.yml`:
```yaml
ports:
  - "5001:5000"
```

### Data persistence

MLflow metadata is stored in PostgreSQL, and artifacts are stored locally inside the container. To persist them across container restarts, consider mapping volumes.

---

