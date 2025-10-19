# Airflow Lab - Regression Model Pipeline

---

## What is Airflow?

Apache Airflow is an open-source platform designed to programmatically author, schedule, and monitor workflows. It enables the orchestration of complex computational workflows and data processing pipelines using Python scripts. Workflows in Airflow are defined as **Directed Acyclic Graphs (DAGs)**, where each node represents a task, and edges define dependencies between them.

![Airflow Working Screenshot](assets/working_dag.png)

The above image shows the Airflow UI with the DAGs running successfully in the lab environment.

---

## Overview

This lab demonstrates a basic **ML pipeline using Apache Airflow** with Docker. The pipeline trains a **Linear Regression model** on student scores data, saves the trained model, and makes predictions on test data. The lab uses **Docker Compose** to orchestrate Airflow components, Redis, and PostgreSQL for a fully functional Airflow environment.

**Apache Airflow** is a platform to programmatically author, schedule, and monitor workflows (DAGs). In this lab, Airflow manages the training pipeline as tasks executed in sequence.

---

## Requirements

To run this lab, you need the following installed on your system:

- **Docker Desktop** (for Windows/macOS) or **Docker Engine** (Linux)  
  This is required to run containerized services like Airflow, PostgreSQL, and Redis without manually installing them.

- **Docker Compose**  
  Used to orchestrate multiple Docker containers together, allowing Airflow and its dependencies to start as a single system.

- **Minimum system resources recommended:**
  - 4 GB RAM
  - 2 CPUs
  - 10 GB disk space

> **Note:** Airflow itself runs inside Docker containers, so you do **not** need to install Python or Airflow locally. Everything is contained in the Docker environment.

---

## Setup Instructions

### Clone the repository
```bash
git clone <your-repo-url>
cd Abhinav_Airflow_Lab2
```

---

## Project Structure
```
.
├── dags/
│   └── regression_pipeline.py      # Airflow DAG defining the workflow
├── docker-compose.yml               # Docker Compose for Airflow, Postgres, Redis
├── Dockerfile                       # Custom Airflow image (optional)
├── plugins/                         # Airflow custom plugins (if any)
├── working_data/
│   ├── student_scores_train.csv    # Training dataset
│   └── student_scores_test.csv     # Test dataset
├── model/
│   └── regression_model.pkl        # Trained model output
└── src/
    └── lab.py                       # Python code for data loading, preprocessing, training, and prediction
```

---

## Docker Setup

This project uses **Docker Compose** to run:

- `airflow-webserver`: Airflow UI at http://localhost:8080
- `airflow-scheduler`: Schedules DAG tasks
- `airflow-worker`: Executes DAG tasks
- `airflow-triggerer`: Handles triggerer jobs for deferrable tasks
- `postgres`: PostgreSQL database for Airflow metadata
- `redis`: Broker for CeleryExecutor

### Build and Run
```bash
# Build and start all services
docker-compose up --build
```

**What happens:**

- `docker-compose up` starts all the services defined in `docker-compose.yml` (Airflow webserver, scheduler, worker, PostgreSQL, Redis, etc.)
- `--build` ensures Docker rebuilds the images if there are any changes to the Dockerfile

### Stop and Remove Containers
```bash
docker-compose down
```

**What happens:**

- Stops all running containers
- Removes networks and temporary containers
- **Important:** It does not remove Docker images by default, so rebuilding is faster next time

### Access the Airflow UI

Open **http://localhost:8080** in your browser to view DAGs, task logs, and the pipeline execution.

---

## Docker Services

- **airflow-webserver**: Airflow UI at http://localhost:8080
- **airflow-scheduler**: Schedules DAG tasks
- **airflow-worker**: Executes DAG tasks
- **airflow-triggerer**: Handles triggerer jobs for deferrable tasks
- **postgres**: PostgreSQL database for Airflow metadata
- **redis**: Broker for CeleryExecutor

---

