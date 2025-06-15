import pandas as pd
import numpy as np
import torch
import pickle
import sys
import os

# Add parent directory to path to import pgc_data_lib if needed
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from pgc_data_lib.metadata import create_metadata, validate_classification_labels, save_dataset_with_metadata

# Load CSV
df = pd.read_csv('data/BENCHMARKS/0001-iris/iris_augmented.csv')

# Features: all columns except 'target', Labels: 'target'
features = df.drop('target', axis=1).values.astype(np.float32)
labels = df['target'].values.astype(np.int64)

# Convert to torch tensors
features_tensor = torch.from_numpy(features)
labels_tensor = torch.from_numpy(labels)

# Create metadata using the library function
metadata = create_metadata(
    features=features_tensor,
    labels=labels_tensor,
    dataset_name='iris-augmented',
    task_type='classification'
)

# Validate label consistency
validate_classification_labels(metadata)

base_filename = '001-iris-augmented.pkl'

# Save dataset with metadata
save_dataset_with_metadata(base_filename, features_tensor, labels_tensor, metadata)
