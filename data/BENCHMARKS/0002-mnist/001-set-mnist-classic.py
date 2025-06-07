import tensorflow_datasets as tfds
import numpy as np
import pickle
import torch

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

# Save as pickle file compatible with dataset_viewer.py
with open('mnist-classic.pkl', 'wb') as f:
    pickle.dump({'features': features, 'labels': labels}, f)

print('Saved mnist-classic.pkl with', features.shape[0], 'samples.')
