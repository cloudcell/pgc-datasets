import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Extract feature columns (excluding 'target')
feature_cols = iris.feature_names
X = df[feature_cols].values

# Compute L2 norms
norms = np.linalg.norm(X, axis=1)

# Normalize features
X_normalized = X / norms[:, np.newaxis]

# Create new dataframe with normalized features
df_normalized = pd.DataFrame(X_normalized, columns=feature_cols)

# Add the norm as a new feature column
df_normalized['norm'] = norms

# Add target column
df_normalized['target'] = df['target']

# Save to CSV
csv_path = "iris_augmented.csv"
df_normalized.to_csv(csv_path, index=False)
print(f"Iris dataset with normalized features and norm saved to {csv_path}")
