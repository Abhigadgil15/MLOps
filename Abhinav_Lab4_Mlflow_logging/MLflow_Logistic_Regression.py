# Block 3: Logistic Regression with MLflow (Fixed Convergence Issue)
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set experiment
mlflow.set_experiment("Iris_Logistic_Regression")

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale the data to improve convergence
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set hyperparameters
max_iter = 1000  # Increased from 1 to fix convergence
penalty = 'l2'
C = 1.0  # Regularization strength (inverse of learning rate)

# Start an MLflow run
with mlflow.start_run(run_name="logistic_regression_scaled"):
    # Log hyperparameters
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("penalty", penalty)
    mlflow.log_param("C", C)
    mlflow.log_param("solver", "lbfgs")
    mlflow.log_param("data_scaled", True)

    # Initialize and train the model
    model = LogisticRegression(
        penalty=penalty, 
        C=C,
        solver='lbfgs', 
        max_iter=max_iter,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_train_pred = model.predict(X_train_scaled)
    
    # Calculate metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)

    # Log metrics
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("train_accuracy", train_accuracy)
    mlflow.log_metric("overfitting_gap", train_accuracy - test_accuracy)

    # Create confusion matrix visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()

    # Save the model
    mlflow.sklearn.log_model(model, "logistic_regression_model")
    
    # Save the scaler as well
    import joblib
    joblib.dump(scaler, "scaler.pkl")
    mlflow.log_artifact("scaler.pkl")

    # Log classification report
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    print("=" * 60)
    print("LOGISTIC REGRESSION RESULTS")
    print("=" * 60)
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    print("=" * 60)
    print("✓ Run complete! Check MLflow UI at http://localhost:5000")