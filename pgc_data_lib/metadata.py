"""
Metadata generation and validation utilities for PGC datasets.
"""
import torch
import numpy as np

# Current version of the dataset format
DATASET_FORMAT_VERSION = "2025-06-15-1434"

# assert DATASET_FORMAT_VERSION is a string of length 15
assert len(DATASET_FORMAT_VERSION) == 15, "DATASET_FORMAT_VERSION must be a string of length 15"

def create_metadata(features, labels, dataset_name, task_type='classification', feature_dim=None, dataset_format_version=DATASET_FORMAT_VERSION):
    """
    Create metadata dictionary for a dataset.
    
    Parameters:
    -----------
    features : torch.Tensor or numpy.ndarray
        The feature data
    labels : torch.Tensor or numpy.ndarray
        The label data
    dataset_name : str
        Name of the dataset
    task_type : str
        Type of task, either 'classification' or 'regression'
    feature_dim : tuple, optional
        Original dimensions of features (e.g., image dimensions)
    dataset_format_version : str, optional
        Version of the dataset format (defaults to DATASET_FORMAT_VERSION)
        
    Returns:
    --------
    dict
        Metadata dictionary containing dataset information
    """
    # Convert numpy arrays to torch tensors if needed
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
        
    # Create metadata dictionary
    metadata = {
        # Dataset format information
        'dataset_format_version': dataset_format_version,
        
        # Dimensionality/feature information
        'num_features': features.shape[1] if len(features.shape) > 1 else 1,
        'feature_dim': feature_dim,
        
        # Task type
        'task_type': task_type,
        'dataset_name': dataset_name
    }
    
    # Add task-specific metadata
    if task_type == 'classification':
        metadata.update({
            'num_classes': len(torch.unique(labels)),
            'min_label': int(torch.min(labels).item()),
            'max_label': int(torch.max(labels).item()),
        })
    elif task_type == 'regression':
        metadata.update({
            'min_label': float(torch.min(labels).item()),
            'max_label': float(torch.max(labels).item()),
            'mean_label': float(torch.mean(labels).item()),
            'std_label': float(torch.std(labels).item()),
        })
    
    return metadata


def validate_classification_labels(metadata):
    """
    Validate that classification labels follow the expected pattern:
    - min_label should be 0
    - max_label should be num_classes - 1
    - max_label - min_label + 1 should equal num_classes
    
    Parameters:
    -----------
    metadata : dict
        Metadata dictionary containing label information
        
    Raises:
    -------
    AssertionError
        If any validation check fails
    """
    assert metadata['task_type'] == 'classification', "This validation is only for classification tasks"
    
    assert metadata['min_label'] == 0, \
        f"Minimum label should be 0, but got {metadata['min_label']}"
    
    assert metadata['max_label'] == metadata['num_classes'] - 1, \
        f"Maximum label should be {metadata['num_classes'] - 1}, but got {metadata['max_label']}"
    
    assert metadata['max_label'] - metadata['min_label'] + 1 == metadata['num_classes'], \
        f"Number of classes ({metadata['num_classes']}) should equal " \
        f"max_label - min_label + 1 ({metadata['max_label'] - metadata['min_label'] + 1})"


def save_dataset_with_metadata(filename, features, labels, metadata):
    """
    Save dataset with metadata to a pickle file.
    
    Parameters:
    -----------
    filename : str
        Path to save the pickle file
    features : torch.Tensor or numpy.ndarray
        The feature data
    labels : torch.Tensor or numpy.ndarray
        The label data
    metadata : dict
        Metadata dictionary containing dataset information
    """
    import pickle
    import time
    
    # if filename already has .pkl extension, remove it
    if filename.endswith('.pkl'):
        filename = filename[:-4]

    # if the prefix is "dataset", remove it
    if filename.startswith('dataset_'):
        filename = filename[8:]
    
    # add suffix to filename as yyyymmdd_hhmmss
    filename = "dataset_" + filename + '_' + time.strftime('%Y%m%d_%H%M%S') + '.pkl'


    # Convert numpy arrays to torch tensors if needed
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)
    
    # Save as pickle file
    with open(filename, 'wb') as f:
        pickle.dump({
            'features': features,
            'labels': labels,
            'metadata': metadata
        }, f)
    
    print(f'Saved {filename} with {features.shape[0]} samples.')
    print('Metadata:')
    for key, value in metadata.items():
        if isinstance(value, dict) and len(value) > 10:
            print(f'  {key}: {type(value)} with {len(value)} items')
        else:
            print(f'  {key}: {value}')

    return metadata

    