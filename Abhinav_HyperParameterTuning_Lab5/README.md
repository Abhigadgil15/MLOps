# Keras Tuner - Hyperparameter Optimization Pipeline

A complete end-to-end machine learning pipeline demonstrating hyperparameter tuning using Keras Tuner on the Diabetes dataset for regression tasks.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [What This Code Does](#what-this-code-does)
- [Hyperparameters Being Tuned](#hyperparameters-being-tuned)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Output Files](#output-files)
- [Understanding the Results](#understanding-the-results)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## 🎯 Overview

This project demonstrates a comprehensive workflow for hyperparameter optimization using **Keras Tuner**. It automates the process of finding the best neural network architecture and training parameters for a regression problem.

**Key Features:**
- Automated hyperparameter search using RandomSearch
- Data preprocessing with StandardScaler
- Model evaluation with multiple metrics (MSE, RMSE, MAE, R²)
- Comprehensive visualizations of model performance
- Training history plots
- Model and hyperparameter persistence

---

## 🔧 Prerequisites

### System Requirements
- **Python Version:** 3.9, 3.10, 3.11, or 3.12
- **Operating System:** macOS, Linux, or Windows
- **RAM:** Minimum 4GB (8GB recommended)

### Required Knowledge
- Basic understanding of Python
- Familiarity with machine learning concepts
- Basic knowledge of neural networks (helpful but not required)

---

## 📦 Installation

### Step 1: Set Up Python Environment

**Option A: Using Virtual Environment (Recommended)**

```bash
# Create virtual environment
python3.11 -m venv myenv

# Activate virtual environment
# On macOS/Linux:
source myenv/bin/activate

# On Windows:
myenv\Scripts\activate
```

**Option B: Using Conda**

```bash
# Create conda environment
conda create -n keras_tuner python=3.11

# Activate environment
conda activate keras_tuner
```

### Step 2: Install Required Packages

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install tensorflow==2.19.1
pip install keras-tuner
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn

# Fix protobuf version (if warnings appear)
pip install protobuf==4.25.3
```

### Step 3: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import keras_tuner as kt; print('Keras Tuner:', kt.__version__)"
```

**Expected Output:**
```
TensorFlow: 2.19.1
Keras Tuner: 1.4.x
```

---

## 📊 Dataset

### Diabetes Dataset (sklearn)

This project uses the built-in **Diabetes dataset** from scikit-learn:

- **Samples:** 442 patients
- **Features:** 10 baseline variables (age, sex, BMI, blood pressure, and 6 blood serum measurements)
- **Target:** Quantitative measure of disease progression one year after baseline
- **Task:** Regression (predicting continuous values)

**Feature Information:**
1. Age
2. Sex
3. Body Mass Index (BMI)
4. Average Blood Pressure
5. S1 - S6: Six blood serum measurements

No manual data download required - the dataset is automatically loaded via `sklearn.datasets.load_diabetes()`.

---

## 🚀 What This Code Does

### Complete Pipeline Overview

This pipeline performs 12 major steps:

#### 1. **Data Loading & Exploration**
   - Loads the diabetes dataset
   - Displays dataset shape and statistics
   - Shows feature names and target distribution

#### 2. **Data Preprocessing**
   - Splits data into training (80%) and testing (20%) sets
   - Applies StandardScaler for feature normalization
   - Prepares data for neural network input

#### 3. **Model Builder Definition**
   - Creates a flexible function that builds neural networks with tunable parameters
   - Defines the search space for hyperparameters

#### 4. **Tuner Initialization**
   - Sets up Keras Tuner's RandomSearch algorithm
   - Configures search parameters (max trials, objective function)

#### 5. **Hyperparameter Search**
   - Systematically tests different hyperparameter combinations
   - Uses early stopping and learning rate reduction callbacks
   - Tracks validation loss to find optimal configuration

#### 6. **Best Model Retrieval**
   - Extracts the best performing hyperparameters
   - Retrieves the best trained model
   - Displays model architecture summary

#### 7. **Model Evaluation**
   - Tests model on unseen test data
   - Calculates performance metrics (MSE, RMSE, MAE, R²)
   - Generates predictions

#### 8. **Top Models Summary**
   - Shows the top 5 best performing models
   - Compares their hyperparameters and performance

#### 9. **Results Visualization**
   - **Actual vs Predicted Plot:** Shows prediction accuracy
   - **Residual Plot:** Identifies prediction patterns/biases
   - **Residual Distribution:** Checks for normality assumptions
   - **Error Metrics Bar Chart:** Visual comparison of different metrics

#### 10. **Sample Predictions Display**
   - Shows first 10 predictions with actual values
   - Calculates individual prediction errors

#### 11. **Final Model Training**
   - Retrains model with best hyperparameters
   - Plots training history (loss and MAE over epochs)
   - Visualizes learning curves

#### 12. **Model Persistence**
   - Saves trained model as `.keras` file
   - Exports best hyperparameters as JSON
   - Saves all visualizations as PNG files

---

## ⚙️ Hyperparameters Being Tuned

The pipeline automatically searches for the best values across these hyperparameters:

| Hyperparameter | Search Space | Description |
|----------------|--------------|-------------|
| **Number of Layers** | 1 - 4 | Number of hidden layers in the network |
| **Units per Layer** | 32 - 256 (step: 32) | Number of neurons in each hidden layer |
| **Dropout Rate** | 0.0 - 0.5 (step: 0.1) | Dropout probability for regularization |
| **Learning Rate** | [0.01, 0.001, 0.0001] | Optimizer step size |
| **Optimizer** | [Adam, RMSprop] | Optimization algorithm |

**Total Search Space:** ~15,000+ possible combinations  
**Trials Tested:** 15 (configurable)

---

## 📁 Project Structure

```
project_directory/
│
├── keras_tuner_pipeline.py          # Main Python script
├── README.md                         # This file
│
├── keras_tuner_results/             # Generated directory
│   └── diabetes_regression/         # Tuner logs and trials
│       ├── trial_*/                 # Individual trial results
│       └── tuner*.json              # Tuner state files
│
├── best_diabetes_model.keras        # Saved best model
├── best_hyperparameters.json        # Best hyperparameters
├── keras_tuner_results.png          # Performance visualizations
└── training_history.png             # Training curves
```

---

## 💻 Usage

### Running the Complete Pipeline

```bash
# Make sure your virtual environment is activated
source myenv/bin/activate  # macOS/Linux
# or
myenv\Scripts\activate     # Windows

# Run the script
python keras_tuner_pipeline.py
```

### Expected Runtime
- **Search Phase:** 3-10 minutes (depending on hardware)
- **Total Pipeline:** 5-15 minutes

### Using in Jupyter Notebook

```python
# Copy the entire code into a Jupyter notebook cell
# Or import and run specific sections

# Example:
%run keras_tuner_pipeline.py
```

---

## 📤 Output Files

After successful execution, you'll find these files:

### 1. `best_diabetes_model.keras`
- **Type:** Keras SavedModel format
- **Size:** ~500 KB - 2 MB
- **Usage:** Load for inference or further training
```python
from tensorflow import keras
model = keras.models.load_model('best_diabetes_model.keras')
predictions = model.predict(new_data)
```

### 2. `best_hyperparameters.json`
- **Type:** JSON configuration file
- **Contains:** All optimal hyperparameter values
- **Example:**
```json
{
    "num_layers": 3,
    "optimizer": "adam",
    "learning_rate": 0.001,
    "units_0": 128,
    "dropout_0": 0.2,
    "units_1": 96,
    "dropout_1": 0.1
}
```

### 3. `keras_tuner_results.png`
- **Type:** High-resolution visualization (300 DPI)
- **Contains:** 4 subplots
  - Actual vs Predicted scatter plot
  - Residual plot
  - Residual distribution histogram
  - Error metrics bar chart

### 4. `training_history.png`
- **Type:** Training progress visualization
- **Contains:** 2 subplots
  - Training/Validation Loss over epochs
  - Training/Validation MAE over epochs

### 5. `keras_tuner_results/` directory
- **Contains:** All trial data and search history
- **Size:** Can be 100+ MB
- **Note:** Can be deleted after training to save space

---

## 📈 Understanding the Results

### Performance Metrics Explained

#### 1. **MSE (Mean Squared Error)**
- **Lower is better**
- Penalizes large errors heavily
- Units: Squared target units
- **Good MSE:** < 3000 for this dataset

#### 2. **RMSE (Root Mean Squared Error)**
- **Lower is better**
- Same units as target variable
- Easier to interpret than MSE
- **Good RMSE:** < 55 for this dataset

#### 3. **MAE (Mean Absolute Error)**
- **Lower is better**
- Average absolute difference
- Less sensitive to outliers than RMSE
- **Good MAE:** < 45 for this dataset

#### 4. **R² Score (Coefficient of Determination)**
- **Range:** -∞ to 1.0 (higher is better)
- Proportion of variance explained
- **Good R²:** > 0.4 for this dataset
- **Excellent R²:** > 0.5

### Interpreting Visualizations

#### Actual vs Predicted Plot
- Points close to the diagonal line = good predictions
- Scatter above line = underestimation
- Scatter below line = overestimation

#### Residual Plot
- Random scatter = good model
- Patterns = model missing something
- Points far from zero = poor predictions

#### Residual Distribution
- Bell-shaped curve = model assumptions met
- Skewed distribution = potential bias

---

## 🎨 Customization

### Using Your Own Dataset

Replace the data loading section:

```python
# Original
data = load_diabetes()
X, y = data.data, data.target

# Your dataset
import pandas as pd
df = pd.read_csv('your_data.csv')
X = df.drop('target_column', axis=1).values
y = df['target_column'].values
```

### Adjusting Hyperparameter Search Space

Modify the `build_model()` function:

```python
# More layers
num_layers = hp.Int('num_layers', min_value=1, max_value=6)

# Different unit ranges
units = hp.Int(f'units_{i}', min_value=64, max_value=512, step=64)

# Additional optimizers
optimizer_name = hp.Choice('optimizer', values=['adam', 'rmsprop', 'sgd'])

# Different learning rates
learning_rate = hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
```

### Using Different Tuner Algorithms

#### Bayesian Optimization (Smarter Search)
```python
tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=20,
    directory='keras_tuner_results',
    project_name='diabetes_bayesian'
)
```

#### Hyperband (Efficient Search)
```python
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=100,
    factor=3,
    directory='keras_tuner_results',
    project_name='diabetes_hyperband'
)
```

### Adjusting Number of Trials

```python
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=30,  # Increase for better results (longer runtime)
    ...
)
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### Issue 1: TensorFlow Installation Fails
```bash
# Error: Could not find a version that satisfies the requirement tensorflow

# Solution: Check Python version
python --version  # Must be 3.9-3.12

# Try installing specific version
pip install tensorflow==2.15.0
```

#### Issue 2: Protobuf Version Warnings
```bash
# Warning: Protobuf gencode version mismatch

# Solution: Downgrade protobuf
pip install protobuf==4.25.3
```

#### Issue 3: Memory Errors During Training
```python
# Error: ResourceExhaustedError: OOM when allocating tensor

# Solution 1: Reduce batch size
tuner.search(..., batch_size=16)  # Instead of 32

# Solution 2: Reduce max_trials
max_trials=10  # Instead of 15

# Solution 3: Limit model size
units = hp.Int(f'units_{i}', min_value=32, max_value=128)  # Smaller range
```

#### Issue 4: Keras Tuner Directory Errors
```bash
# Error: Permission denied when creating directory

# Solution: Use absolute path
import os
tuner_dir = os.path.join(os.getcwd(), 'keras_tuner_results')
tuner = kt.RandomSearch(..., directory=tuner_dir)
```

#### Issue 5: Plots Not Displaying
```python
# Issue: Plots don't show in terminal

# Solution: Save to file and view separately
plt.savefig('plot.png')
# Then open plot.png
```

#### Issue 6: Slow Training on Mac
```python
# Issue: Training is very slow on macOS

# Check if using Metal (Apple GPU acceleration)
import tensorflow as tf
print("GPU Available:", len(tf.config.list_physical_devices('GPU')))

# If not available, install tensorflow-metal
pip install tensorflow-metal
```

---

## 📚 References

### Documentation
- [Keras Tuner Official Documentation](https://keras.io/keras_tuner/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Scikit-learn Datasets](https://scikit-learn.org/stable/datasets.html)

### Tutorials
- [Keras Tuner Getting Started](https://keras.io/guides/keras_tuner/getting_started/)
- [Hyperparameter Tuning Best Practices](https://www.tensorflow.org/tutorials/keras/keras_tuner)

### Papers
- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560)
- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)

---

## 📝 License

This project is provided as-is for educational purposes.

---

## 👨‍💻 Author

Abhinav Gadgil  
MS Computer Science Student | Northeastern University

---

## 🤝 Contributing

Feel free to fork this project and adapt it for your own use cases. Suggestions and improvements are welcome!

---

## 📧 Support

If you encounter any issues:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review TensorFlow/Keras Tuner documentation
3. Check Python and package versions

---

## ⭐ Acknowledgments

- Scikit-learn team for the Diabetes dataset
- TensorFlow and Keras teams for the deep learning framework
- Keras Tuner developers for the hyperparameter optimization library

---

**Last Updated:** November 2025  
**Version:** 1.0.0
