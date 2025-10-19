import mlflow

# Point MLflow to your local server
mlflow.set_tracking_uri("http://localhost:5050")
mlflow.set_experiment("basic-demo")

with mlflow.start_run(run_name="test-run"):
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("loss", 0.2)

print("✅ Logged parameters and metrics to MLflow!")
