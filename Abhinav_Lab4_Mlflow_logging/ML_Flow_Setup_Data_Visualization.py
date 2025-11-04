# Block 1: Setup and Initial Data Visualization
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for easier visualization
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['species'] = pd.Categorical.from_codes(y, iris.target_names)

# Visualization 1: Feature Distribution
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Iris Dataset - Feature Distributions', fontsize=16)

for idx, feature in enumerate(iris.feature_names):
    row = idx // 2
    col = idx % 2
    for species in iris.target_names:
        data = iris_df[iris_df['species'] == species][feature]
        axes[row, col].hist(data, alpha=0.5, label=species, bins=15)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Frequency')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 2: Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = iris_df.iloc[:, :-1].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Visualization 3: Pairplot
plt.figure(figsize=(12, 10))
pairplot = sns.pairplot(iris_df, hue='species', palette='Set2', 
                        diag_kind='kde', plot_kws={'alpha': 0.6})
pairplot.fig.suptitle('Iris Dataset - Pairwise Feature Relationships', 
                      y=1.02, fontsize=16)
plt.savefig('pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# Dataset Statistics
print("=" * 60)
print("IRIS DATASET STATISTICS")
print("=" * 60)
print(f"\nDataset Shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
print("\nFeature Statistics:")
print(iris_df.describe())
print("=" * 60)