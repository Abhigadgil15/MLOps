# Block 6: GridSearchCV with MLflow (Corrected and Enhanced)
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set experiment name in MLflow
mlflow.set_experiment("GridSearch_RandomForest")

# Load the dataset
data = load_iris()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Set up the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Define the parameters for the GridSearch
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 5, 10]
}

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=rf, 
    param_grid=param_grid, 
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

# Start the parent MLflow run
with mlflow.start_run(run_name="gridsearch_parent") as parent_run:
    # Run GridSearch
    print("=" * 60)
    print("STARTING GRID SEARCH")
    print("=" * 60)
    grid_search.fit(X_train, y_train)
    
    # Get results
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Log each trial as a nested run
    for i in range(len(results_df)):
        with mlflow.start_run(run_name=f"trial_{i}", nested=True):
            params = results_df.loc[i, 'params']
            mean_score = results_df.loc[i, 'mean_test_score']
            std_score = results_df.loc[i, 'std_test_score']
            
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            mlflow.log_metric("mean_cv_accuracy", mean_score)
            mlflow.log_metric("std_cv_accuracy", std_score)
    
    # Log best parameters to parent run
    for key, value in grid_search.best_params_.items():
        mlflow.log_param(f"best_{key}", value)
    
    # Log best score
    mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)
    
    # Test the best model
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_accuracy)
    
    # Log the best model
    mlflow.sklearn.log_model(best_model, "best_random_forest_model")
    
    # Create visualizations
    # 1. GridSearch Results Heatmap
    pivot_table = results_df.pivot_table(
        values='mean_test_score',
        index='param_max_depth',
        columns='param_n_estimators',
        aggfunc='max'
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='YlGnBu')
    plt.title('GridSearch Results: Max Accuracy by Depth and Estimators')
    plt.xlabel('Number of Estimators')
    plt.ylabel('Max Depth')
    plt.tight_layout()
    plt.savefig('gridsearch_heatmap.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('gridsearch_heatmap.png')
    plt.close()
    
    # 2. Feature Importance
    feature_importance = pd.DataFrame({
        'feature': data.feature_names,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance - Best Random Forest Model')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('feature_importance.png')
    plt.close()
    
    # 3. Confusion Matrix
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=data.target_names,
                yticklabels=data.target_names)
    plt.title('Confusion Matrix - Best Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('rf_confusion_matrix.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('rf_confusion_matrix.png')
    plt.close()
    
    # Save detailed results
    results_df.to_csv('gridsearch_results.csv', index=False)
    mlflow.log_artifact('gridsearch_results.csv')
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=data.target_names)
    with open("rf_classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("rf_classification_report.txt")
    
    # Print results
    print("\n" + "=" * 60)
    print("GRID SEARCH RESULTS")
    print("=" * 60)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"\nClassification Report:\n{report}")
    print("=" * 60)
    print("✓ Grid search complete! Check MLflow UI at http://localhost:5000")