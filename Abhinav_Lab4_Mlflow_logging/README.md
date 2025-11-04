🧪 MLflow Experiment Tracking Guide

A comprehensive guide demonstrating MLflow’s experiment tracking capabilities with various machine learning models.
This project covers everything from basic logging to advanced hyperparameter tuning with GridSearchCV.

📋 Prerequisites

Install the required dependencies:

pip install mlflow scikit-learn tensorflow matplotlib seaborn pandas numpy

🚀 Quick Start
Running the Experiments

Execute each block sequentially to build your understanding step by step.

Block 1: Setup and Visualizations
python MLflow_Setup_and_Basic_Stats.py


Generates dataset statistics, feature distributions, correlation heatmaps, and pairwise relationships for the Iris dataset.

Block 2: Basic MLflow Tracking
python MLflow_Basic_Tracking.py


Demonstrates fundamental MLflow concepts: logging parameters, metrics, and artifacts over multiple steps.

Block 3: Logistic Regression with Scaling
python MLflow_Logistic_Regression.py


Runs a full logistic regression pipeline with data scaling, confusion matrix visualization, and comprehensive metrics logging.

Block 4: Autologging Example
python MLflow_Autologging.py


Shows MLflow’s autologging feature that automatically captures model parameters, metrics, and artifacts with minimal code.

Block 5: Keras Neural Network
python MLflow_Keras_MNIST.py


Deep learning example using TensorFlow/Keras on MNIST dataset. Demonstrates autologging for neural networks and training visualization.

Block 6: GridSearch with Random Forest
python MLflow_GridSearch_RF.py


Advanced hyperparameter tuning using GridSearchCV with nested MLflow runs.
Includes heatmaps, feature importance, and detailed performance metrics across trials.

📊 Viewing Results

Start the MLflow UI to visualize and compare experiments:

mlflow ui


Access at 👉 http://localhost:5000

🔧 Key Improvements
🧮 Convergence Fixes

Increased max_iter in LogisticRegression from 1 → 1000

Applied StandardScaler for normalization

Configured solver to eliminate convergence warnings

💾 Model Persistence

Dynamic run_id retrieval for model loading

Correct URI format: runs:/{run_id}/model

Saved scalers alongside models for reproducibility

⚙️ Modern API Usage

Updated Keras to use Input() instead of deprecated input_shape

Full compatibility with TensorFlow 2.x and Keras 3.x

Suppressed version compatibility warnings

📈 Enhanced Logging

Nested runs for GridSearch with parent-child hierarchy

Logged all visualizations and artifacts

Tracked both train/test accuracy and overfitting gaps

📊 Visualizations Generated
Category	Visualizations
Feature Analysis	Distributions, correlation matrices, pairplots
Model Performance	Confusion matrices, classification reports
Training Progress	Accuracy/Loss curves for neural networks
Hyperparameter Tuning	GridSearch heatmaps, parameter importance
Feature Engineering	Feature importance rankings
📁 Project Structure
mlflow-experiments/
├── mlruns/                         # Auto-generated MLflow tracking directory
│   ├── 0/                          # Default experiment
│   ├── .trash/                     # Deleted runs
│   └── experiments/                # Metadata
│
├── artifacts/                      # Generated visualizations and models
│   ├── feature_distributions.png
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── gridsearch_heatmap.png
│
├── MLflow_Setup_and_Basic_Stats.py     # Dataset exploration
├── MLflow_Basic_Tracking.py            # Basic tracking demo
├── MLflow_Logistic_Regression.py       # Classification with scaling
├── MLflow_Autologging.py               # Autologging demonstration
├── MLflow_Keras_MNIST.py               # Deep learning example
├── MLflow_GridSearch_RF.py             # GridSearch optimization
└── README.md

💡 MLflow CLI Commands
# Start MLflow UI (default port)
mlflow ui

# Start on custom port
mlflow ui --port 5001

# Use specific backend storage
mlflow ui --backend-store-uri ./mlruns

# List all experiments
mlflow experiments list

# Search runs by experiment ID
mlflow runs list --experiment-id 1

# Delete experiment (moves to .trash)
mlflow experiments delete --experiment-id <ID>

# Restore deleted experiment
mlflow experiments restore --experiment-id <ID>

📈 Tips for Effective Demonstrations

✅ Run blocks sequentially — builds from simple to advanced
✅ Keep MLflow UI open — watch live updates as models train
✅ Compare runs — analyze performance across models
✅ Export charts — useful for reports or presentations
✅ Explore nested runs — in GridSearch to visualize tuning
✅ Manage artifacts — all results versioned and retrievable

🔍 Block Overview
Block	Concept	Key Takeaway
1	Data Exploration	Understand dataset before modeling
2	Basic Tracking	Manual logging of params/metrics/artifacts
3	Full Pipeline	Preprocessing + training + evaluation
4	Autologging	Minimal code, maximum tracking
5	Deep Learning	Neural network tracking with Keras
6	Optimization	Systematic hyperparameter search
🛠️ Troubleshooting

Port already in use

mlflow ui --port 5001


Cannot find experiment

mlflow experiments list


Model loading fails

Use correct run_id from MLflow UI

Ensure artifact path is model/ not models/

Import errors

pip list | grep mlflow


Use a virtual environment if conflicts arise.

📚 Additional Resources

MLflow Documentation

MLflow Tracking API

Scikit-learn Integration

TensorFlow/Keras Integration

🧠 Note:
The mlruns/ directory is automatically created when you run your first experiment.
It stores all experiment metadata, metrics, parameters, and artifacts by default.