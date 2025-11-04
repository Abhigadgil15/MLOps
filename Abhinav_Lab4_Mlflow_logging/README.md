# 🚀 MLflow Experiment Tracking Guide

> A comprehensive guide demonstrating MLflow's experiment tracking capabilities with machine learning models from basic tracking to advanced hyperparameter tuning.

---

## 📋 Prerequisites

Install required dependencies:

```bash
pip install mlflow scikit-learn tensorflow matplotlib seaborn pandas numpy
```

---

## 🎯 Quick Start

### Running the Experiments

Execute the code blocks in sequence:

### 📊 Block 1: Setup and Visualizations

```bash
python MLflow_Setup_and_Basic_Stats.py
```

**What it does:** 

Generates dataset statistics, feature distributions, correlation heatmaps, and pairwise relationships for the Iris dataset.

---

### 🔍 Block 2: Basic MLflow Tracking

```bash
python MLflow_Basic_Tracking.py
```

**What it does:** 

Demonstrates fundamental MLflow concepts - logging parameters, metrics over multiple steps, and artifacts.

---

### 🎓 Block 3: Logistic Regression with Scaling

```bash
python MLflow_Logistic_Regression.py
```

**What it does:** 

Complete logistic regression pipeline with data scaling, confusion matrix visualization, and comprehensive metrics logging. 

**Fixes convergence issues** through proper preprocessing.

---

### ⚡ Block 4: Autologging Example

```bash
python MLflow_Autologging.py
```

**What it does:** 

Shows MLflow's autologging feature that automatically captures model parameters, metrics, and artifacts with minimal code. 

Includes model loading and inference examples.

---

### 🧠 Block 5: Keras Neural Network (MNIST)

```bash
python MLflow_Keras_MNIST.py
```

**What it does:** 

Deep learning example using TensorFlow/Keras on MNIST dataset. 

Demonstrates autologging for neural networks with training history visualization.

---

### 🔬 Block 6: GridSearch with Random Forest

```bash
python MLflow_GridSearch_RF.py
```

**What it does:** 

Advanced hyperparameter tuning using GridSearchCV with nested runs. 

Includes comprehensive visualizations: heatmaps of parameter combinations, feature importance, and performance metrics.

---

## 🖥️ Viewing Results

Start the MLflow UI to visualize and compare experiments:

```bash
mlflow ui
```

Then open your browser to: **http://localhost:5000**

---

## ✨ Key Improvements Made

### 🔧 Convergence Fixes

- ✅ Increased `max_iter` from 1 to 1000 in LogisticRegression

- ✅ Implemented `StandardScaler` for feature normalization

- ✅ Proper solver configuration to prevent convergence warnings

### 💾 Model Persistence

- ✅ Dynamic `run_id` retrieval for model loading

- ✅ Correct URI formatting (`runs:/{run_id}/model`)

- ✅ Scaler artifacts saved alongside models for inference

### 🆕 Modern API Usage

- ✅ Updated Keras to use `Input` layer instead of deprecated `input_shape` parameter

- ✅ Compatible with TensorFlow 2.x and Keras 3.x

- ✅ Proper warning suppression for version compatibility

### 📈 Enhanced Logging

- ✅ Nested runs for GridSearch trials (parent-child relationships)

- ✅ Comprehensive artifact logging (plots, reports, model files)

- ✅ Multi-dimensional metric tracking (train/test accuracy, overfitting gaps)

---

## 📊 Visualizations Generated

Each experiment creates informative visualizations automatically:

| Visualization Type | Description |
|-------------------|-------------|
| 🎨 **Feature Analysis** | Distribution plots, correlation matrices, pairplots |
| 📉 **Model Performance** | Confusion matrices, classification reports |
| 📈 **Training Progress** | Accuracy/loss curves for neural networks |
| 🔥 **Hyperparameter Tuning** | GridSearch heatmaps, parameter importance |
| 🌟 **Feature Engineering** | Feature importance rankings |

---

## 📁 Project Structure

```
mlflow-experiments/
│
├── 📂 mlruns/                           # MLflow tracking directory (auto-generated)
│   ├── 0/                               # Default experiment
│   ├── .trash/                          # Deleted runs
│   └── experiments/                     # Experiment metadata
│
├── 📂 artifacts/                        # Generated visualizations and models
│   ├── feature_distributions.png
│   ├── confusion_matrix.png
│   ├── training_history.png
│   └── gridsearch_heatmap.png
│
├── 📄 MLflow_Setup_and_Basic_Stats.py  # Dataset exploration and stats
├── 📄 MLflow_Basic_Tracking.py         # Simple tracking example
├── 📄 MLflow_Logistic_Regression.py    # Classification with preprocessing
├── 📄 MLflow_Autologging.py            # Automatic logging demo
├── 📄 MLflow_Keras_MNIST.py            # Deep learning example
├── 📄 MLflow_GridSearch_RF.py          # Hyperparameter optimization
│
└── 📖 README.md                         # This file
```

---

## 🛠️ MLflow CLI Commands

Essential commands for managing experiments:

```bash
# 🚀 Start UI on default port
mlflow ui

# 🔌 Start UI on custom port
mlflow ui --port 5001

# 💾 Specify backend storage location
mlflow ui --backend-store-uri ./mlruns

# 📋 List all experiments
mlflow experiments list

# 🔍 Search runs with filters
mlflow runs list --experiment-id 1

# 🗑️ Delete an experiment (moves to .trash)
mlflow experiments delete --experiment-id <ID>

# ♻️ Restore deleted experiment
mlflow experiments restore --experiment-id <ID>
```

---

## 💡 Tips for Effective Demonstrations

**1. 📚 Sequential Execution**

Run blocks 1-6 in order to show progression from basic to advanced concepts

**2. 👀 Real-time Monitoring**

Keep MLflow UI open in browser while running experiments to see live updates

**3. ⚖️ Comparative Analysis**

Use the "Compare" feature in UI to analyze multiple runs side-by-side

**4. 📤 Export Capabilities**

Download charts and reports directly from the UI for presentations

**5. 🔗 Nested Organization**

GridSearch example shows how to structure complex experiments with parent-child runs

**6. 📦 Artifact Management**

All plots and models are versioned and retrievable through the UI

---

## 📚 What Each Block Demonstrates

| Block | 🎯 Concept | 💎 Key Takeaway |
|-------|-----------|----------------|
| **1** | Data Exploration | Understanding dataset before modeling |
| **2** | Basic Tracking | Manual logging of params, metrics, artifacts |
| **3** | Full Pipeline | Preprocessing + training + evaluation |
| **4** | Autologging | Minimal code, maximum tracking |
| **5** | Deep Learning | Neural network tracking with Keras |
| **6** | Optimization | Systematic hyperparameter search with nested runs |

---

## 🔧 Troubleshooting

### ⚠️ Port already in use

```bash
mlflow ui --port 5001  # Use different port
```

### ❓ Cannot find experiment

```bash
mlflow experiments list  # Verify experiment exists
```

### 🚫 Model loading fails

- ✅ Ensure you're using the correct `run_id` from the MLflow UI

- ✅ Check that the artifact path is `model` not `models`

- ✅ Verify the experiment hasn't been deleted

### 📦 Import errors

```bash
# Check installed packages
pip list | grep mlflow

# Use virtual environment to avoid conflicts
python -m venv mlflow_env
source mlflow_env/bin/activate  # On Windows: mlflow_env\Scripts\activate
pip install -r requirements.txt
```

---

## 🌐 Additional Resources

- 📖 [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

- 🔍 [MLflow Tracking API](https://mlflow.org/docs/latest/tracking.html)

- 🤖 [Scikit-learn Integration](https://mlflow.org/docs/latest/python_api/mlflow.sklearn.html)

- 🧠 [TensorFlow/Keras Integration](https://mlflow.org/docs/latest/python_api/mlflow.keras.html)

---

## 📝 Notes

> **💡 Tip:** The `mlruns` directory is created automatically when you run your first experiment. All experiment data, metrics, parameters, and artifacts are stored here by default.

> **⚠️ Warning:** Don't manually edit files in the `mlruns` directory as it may corrupt your experiment data.

---

<div align="center">

### 🎉 Happy Tracking with MLflow! 🎉

Made with ❤️ for ML Engineers

</div>
