import pandas as pd
import numpy as np
import torch
import pickle

# Load CSV
df = pd.read_csv('data/BENCHMARKS/0001-iris/iris_augmented.csv')

# Features: all columns except 'target', Labels: 'target'
features = df.drop('target', axis=1).values.astype(np.float32)
labels = df['target'].values.astype(np.int64)

# Convert to torch tensors
features_tensor = torch.from_numpy(features)
labels_tensor = torch.from_numpy(labels)

# Save all samples in a single pickle file
with open('0011-iris-all.pkl', 'wb') as f:
    pickle.dump({'features': features_tensor, 'labels': labels_tensor}, f)

print(f'Saved 0011-iris-all.pkl with {features.shape[0]} samples.')
