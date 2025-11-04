MLflow Experiment Tracking Guide
A comprehensive guide demonstrating MLflow's experiment tracking capabilities with machine learning models. This project includes multiple examples from basic tracking to advanced hyperparameter tuning with GridSearch.
📋 Prerequisites
Install required dependencies:
bashpip install mlflow scikit-learn tensorflow matplotlib seaborn pandas numpy
🚀 Quick Start
Running the Experiments
Execute the code blocks in sequence to build understanding progressively:
Block 1: Setup and Visualizations
bashpython MLflow_Setup_and_Basic_Stats.py
Generates dataset statistics, feature distributions, correlation heatmaps, and pairwise relationships for the Iris dataset.
Block 2: Basic MLflow Tracking
bashpython MLflow_Basic_Tracking.py
Demonstrates fundamental MLflow concepts: logging parameters, metrics over multiple steps, and artifacts.
Block 3: Logistic Regression with Scaling
bashpython MLflow_Logistic_Regression.py
Complete logistic regression pipeline with data scaling, confusion matrix visualization, and comprehensive metrics logging. Fixes convergence issues through proper preprocessing.
Block 4: Autologging Example
bashpython MLflow_Autologging.py
Shows MLflow's autologging feature that automatically captures model parameters, metrics, and artifacts with minimal code. Includes model loading and inference examples.
Block 5: Keras Neural Network
bashpython MLflow_Keras_MNIST.py
Deep learning example using TensorFlow/Keras on MNIST dataset. Demonstrates autologging for neural networks with training history visualization and model evaluation.
Block 6: GridSearch with Random Forest
bashpython MLflow_GridSearch_RF.py
Advanced hyperparameter tuning using GridSearchCV with nested runs. Includes comprehensive visualizations: heatmaps of parameter combinations, feature importance, and performance metrics across all trials.
Viewing Results
Start the MLflow UI to visualize and compare experiments:
bashmlflow ui
Access the interface at: http://localhost:5000
🔧 Key Improvements
Convergence Fixes

Increased max_iter from 1 to 1000 in LogisticRegression
Implemented StandardScaler for feature normalization
Proper solver configuration to prevent convergence warnings

Model Persistence

Dynamic run_id retrieval for model loading
Correct URI formatting (runs:/{run_id}/model)
Scaler artifacts saved alongside models for inference

Modern API Usage

Updated Keras to use Input layer instead of deprecated input_shape parameter
Compatible with TensorFlow 2.x and Keras 3.x
Proper warning suppression for version compatibility checks

Enhanced Logging

Nested runs for GridSearch trials showing parent-child relationships
Comprehensive artifact logging including plots, reports, and model files
Multi-dimensional metric tracking (train/test accuracy, overfitting gaps)

📊 Visualizations Generated
Each experiment creates informative visualizations automatically:

Feature Analysis: Distribution plots, correlation matrices, pairplots
Model Performance: Confusion matrices, classification reports
Training Progress: Accuracy/loss curves for neural networks
Hyperparameter Tuning: GridSearch heatmaps, parameter importance
Feature Engineering: Feature importance rankings

📁 Project Structure
mlflow-experiments/
├── mlruns/                              # MLflow tracking directory (auto-generated)
│   ├── 0/                               # Default experiment
│   ├── .trash/                          # Deleted runs
│   └── experiments/                     # Experiment metadata
│
├── artifacts/                           # Generated visualizations and models
│   ├── feature_distributions.png
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── gridsearch_heatmap.png
│
├── MLflow_Setup_and_Basic_Stats.py     # Dataset exploration and stats
├── MLflow_Basic_Tracking.py            # Simple tracking example
├── MLflow_Logistic_Regression.py       # Classification with preprocessing
├── MLflow_Autologging.py               # Automatic logging demo
├── MLflow_Keras_MNIST.py               # Deep learning example
├── MLflow_GridSearch_RF.py             # Hyperparameter optimization
│
└── README.md                            # This file
💡 MLflow CLI Commands
Useful commands for managing experiments:
bash# Start UI on default port
mlflow ui

# Start UI on custom port
mlflow ui --port 5001

# Specify backend storage location
mlflow ui --backend-store-uri ./mlruns

# List all experiments
mlflow experiments list

# Search runs with filters
mlflow runs list --experiment-id 1

# Delete an experiment (moves to .trash)
mlflow experiments delete --experiment-id <ID>

# Restore deleted experiment
mlflow experiments restore --experiment-id <ID>
📈 Tips for Effective Demonstrations

Sequential Execution: Run blocks 1-6 in order to show progression from basic to advanced concepts
Real-time Monitoring: Keep MLflow UI open in browser while running experiments to see live updates
Comparative Analysis: Use the "Compare" feature in UI to analyze multiple runs side-by-side
Export Capabilities: Download charts and reports directly from the UI for presentations
Nested Organization: GridSearch example shows how to structure complex experiments with parent-child runs
Artifact Management: All plots and models are versioned and retrievable through the UI

🔍 What Each Block Demonstrates
BlockConceptKey Takeaway1Data ExplorationUnderstanding dataset before modeling2Basic TrackingManual logging of params, metrics, artifacts3Full PipelinePreprocessing + training + evaluation4AutologgingMinimal code, maximum tracking5Deep LearningNeural network tracking with Keras6OptimizationSystematic hyperparameter search with nested runs
🛠️ Troubleshooting
Port already in use:
bashmlflow ui --port 5001  # Use different port
Cannot find experiment:
bashmlflow experiments list  # Verify experiment exists
Model loading fails:

Ensure you're using the correct run_id from the MLflow UI
Check that the artifact path is model not models

Import errors:

Verify all dependencies installed: pip list | grep mlflow
Use virtual environment to avoid conflicts

📚 Additional Resources

MLflow Documentation
MLflow Tracking API
Scikit-learn Integration
TensorFlow/Keras Integration


Note: The mlruns directory is created automatically when you run your first experiment. All experiment data, metrics, parameters, and artifacts are stored here by default.