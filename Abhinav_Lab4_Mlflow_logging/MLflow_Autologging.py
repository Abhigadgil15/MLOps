# Block 4: MLflow Autologging (Corrected)
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import pandas as pd

# Set experiment name
mlflow.set_experiment("Iris_Autologging_Experiment")

# Enable autologging
mlflow.sklearn.autolog()

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a model with autologging
with mlflow.start_run(run_name="autolog_logistic"):
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Autologging handles:
    # - Model parameters
    # - Training metrics
    # - Model signature
    # - Model artifacts
    
    score = model.score(X_test, y_test)
    print(f"Test Accuracy: {score:.4f}")
    print("✓ Autologging complete!")
    
    # Get the run ID for later model loading
    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")

# Model Loading Example
print("\n" + "=" * 60)
print("LOADING AND TESTING MODEL")
print("=" * 60)

# Load model using the run_id
logged_model = f'runs:/{run_id}/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Make predictions
predictions = loaded_model.predict(pd.DataFrame(X_test, 
                                                columns=data.feature_names))
print(f"Predictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:5]}")
print("✓ Model loaded and tested successfully!")