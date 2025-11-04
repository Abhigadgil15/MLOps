MLflow Experiment Tracking Guide
Prerequisites
bashpip install mlflow scikit-learn tensorflow matplotlib seaborn pandas numpy
Running the Code Blocks
1. Setup and Visualizations
Run this first to see dataset statistics and visualizations:
bashpython MLflow_Setup_and_Basic_Stats.py
2. Basic MLflow Tracking
bashpython MLflow_Basic_Tracking.py
3. Logistic Regression (Fixed)
bashpython MLflow_Logistic_Regression.py
4. Autologging Example
bashpython MLflow_Autologging.py
5. GridSearch with Random Forest
bashpython MLflow_GridSearch_RF.py
Starting MLflow UI
After running any experiment, start the MLflow UI to view results:
bashmlflow ui
Then open your browser to: http://localhost:5000
Key Fixes Made

Convergence Issues:

Increased max_iter from 1 to 1000
Added data scaling with StandardScaler
Used proper model configuration


Model Loading:

Fixed the run_id retrieval
Proper model URI formatting
Added error handling


Keras Deprecation:

Used Input layer instead of input_shape parameter
Updated to modern Keras API


GridSearch Logging:

Added nested runs for better organization
Comprehensive visualization of results
Feature importance plots


Added Visualizations:

Feature distributions
Correlation heatmaps
Confusion matrices
Training history plots
GridSearch heatmaps
Feature importance charts



MLflow Commands Reference
bash# Start UI
mlflow ui

# Start UI on different port
mlflow ui --port 5001

# Start UI with specific backend store
mlflow ui --backend-store-uri ./mlruns

# View experiments programmatically
mlflow experiments list

# Delete an experiment
mlflow experiments delete --experiment-id <ID>
Project Structure
your_project/
├── mlruns/                          # MLflow tracking data
├── MLflow_Setup_and_Basic_Stats.py
├── MLflow_Basic_Tracking.py
├── MLflow_Logistic_Regression.py
├── MLflow_Autologging.py
├── MLflow_Keras_MNIST.py
├── MLflow_GridSearch_RF.py
└── artifacts/                       # Generated plots and models
Tips for Demonstration

Run blocks in order to build understanding progressively
Keep MLflow UI open while running experiments
Compare runs using the MLflow UI comparison feature
Export results from the UI for presentations
Use nested runs for complex experiments like GridSearch
