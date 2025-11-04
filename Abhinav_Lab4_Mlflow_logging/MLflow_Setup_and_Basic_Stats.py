# Block 2: Basic MLflow Tracking (Corrected)
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Set experiment name
mlflow.set_experiment("Iris_Basic_Tracking")

# Start an MLflow run
with mlflow.start_run(run_name="basic_simulation"):
    # Log hyperparameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 10)

    # Log metrics (in a loop simulating training)
    for epoch in range(10):
        accuracy = 0.8 + epoch * 0.02  # Simulated accuracy improvement
        mlflow.log_metric("accuracy", accuracy, step=epoch)

    # Log an artifact (a result file)
    with open("output.txt", "w") as f:
        f.write("This is a test artifact from basic tracking experiment.")
    mlflow.log_artifact("output.txt")
    
    print("✓ Basic tracking run complete!")
    print("  - Logged 2 parameters")
    print("  - Logged 10 metric values")
    print("  - Logged 1 artifact")