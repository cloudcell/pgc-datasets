import tensorflow_datasets as tfds
import numpy as np
import pickle
import torch
import sys
import os

# Add parent directory to path to import pgc_data_lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from pgc_data_lib.metadata import create_metadata, validate_classification_labels, save_dataset_with_metadata

# Load Fashion MNIST dataset
(train_ds, test_ds), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

# merge both datasets (because split will be done during the training)
train_ds = train_ds.concatenate(test_ds)

# Convert the tf.data.Dataset to numpy arrays
features = []
labels = []
for image, label in tfds.as_numpy(train_ds):
    features.append(image)
    labels.append(label)

features = np.stack(features)  # shape: (num_samples, 28, 28)
labels = np.array(labels)      # shape: (num_samples,)

# Flatten each image to 1D (28*28 = 784)
features = features.reshape(features.shape[0], -1)  # shape: (num_samples, 784)

# Convert to torch tensors for compatibility with dataset_viewer.py
features = torch.from_numpy(features)
labels = torch.from_numpy(labels)

# Create metadata using the library function
metadata = create_metadata(
    features=features,
    labels=labels,
    dataset_name='mnist-classic',
    task_type='classification',
    feature_dim=(28, 28)  # Original image dimensions
)



base_filename = '005-mnist-classic'

# Save dataset with metadata
save_dataset_with_metadata(base_filename, features, labels, metadata)
