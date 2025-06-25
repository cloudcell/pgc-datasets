#!/usr/bin/env python3
"""
Script to convert arithmetic text datasets to binary format with sliding windows.
Takes the arithmetic datasets from data/ARITHMETIC/maths and creates binary datasets
with a sliding window of 16 characters, moving 1 character at a time.
Each window predicts the next character in the sequence.
"""

import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm
import glob
import argparse

# Add parent directory to path to import pgc_data_lib
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from pgc_data_lib.metadata import create_metadata, validate_classification_labels, save_dataset_with_metadata

# Constants
WINDOW_SIZE = 16  # Size of the sliding window in characters
SLIDE_STEP = 1    # Move the window by 1 character at a time

class ArithmeticBinaryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def char_to_binary(char):
    """Convert a character to its 8-bit binary representation."""
    # Only consider ASCII characters (0-127)
    ascii_val = ord(char) & 127
    return [int(b) for b in format(ascii_val, '08b')]

def process_text_file(file_path):
    """
    Process text file and convert to binary sequences with sliding windows.
    Uses a window of 16 characters and slides by 1 character at a time.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    all_features = []
    all_labels = []
    
    print(f"Processing {file_path}...")
    for line_idx, line in enumerate(tqdm(lines, desc=f"Processing lines in {os.path.basename(file_path)}")):
        # Remove newline character but keep all spaces
        if line.endswith('\n'):
            line = line[:-1]
        
        # We need at least WINDOW_SIZE + 1 characters (window + next character to predict)
        if len(line) < WINDOW_SIZE + 1:
            continue  # Skip lines that are too short
        
        # Process each possible window in the line
        for i in range(len(line) - WINDOW_SIZE):
            # Get the current window
            window = line[i:i+WINDOW_SIZE]
            
            # The next character after our window is the label
            next_char = line[i+WINDOW_SIZE]
            
            # Convert window characters to binary
            window_binary = []
            for char in window:
                window_binary.extend(char_to_binary(char))
            
            # Add to our dataset
            all_features.append(window_binary)
            all_labels.append(ord(next_char) & 127)  # Get ASCII value of next char
    
    return all_features, all_labels

def main():
    # arguments:
    # input: path to the input text file
    # output: path to the output binary file
    # dataset_name: name of the dataset (default: maths-binary)

    parser = argparse.ArgumentParser(description='Convert text dataset to binary format with sliding windows.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input text file')
    # parser.add_argument('--output', type=str, default='dataset_012-maths-binary.pkl', 
    #                     help='Path to the output binary file (default: dataset_012-maths-binary.pkl)')
    parser.add_argument('--dataset-name', type=str, default='maths-binary', 
                        help='Name of the dataset (default: maths-binary)')
    
    args = parser.parse_args()
    
    # Input and output paths
    input_file = args.input
    # output_file = args.output
    dataset_name = args.dataset_name
    
    if not input_file:
        print(f"No text files found in {input_file}")
        return
    
    # Process the text file
    features, labels = process_text_file(input_file)
        
    if not features:
        print(f"No valid samples generated from {input_file}")
        return
        
    # Create the dataset
    dataset = ArithmeticBinaryDataset(features, labels)
    
    # Create metadata
    feature_dim = (WINDOW_SIZE * 8,)  # 8 bits per character
    num_classes = 256  # ASCII values 0-127
    
    # Create metadata using the library function
    metadata = create_metadata(
        features=dataset.features,
        labels=dataset.labels,
        dataset_name=dataset_name,
        task_type="classification",
        feature_dim=feature_dim,
        num_classes=num_classes
    )
    
    # Validate classification labels
    try:
        validate_classification_labels(metadata)
        print("Label validation passed.")
    except AssertionError as e:
        print(f"Warning: Label validation failed: {e}")
    
    # Save dataset with metadata
    base_filename = 'dataset_012-maths-binary'
    save_dataset_with_metadata(base_filename, dataset.features, dataset.labels, metadata)
    
    print(f"Dataset created for {input_file} with {len(dataset)} samples")
    print(f"Sample entry:")
    print(f"Features shape: {dataset.features[0].shape}")
    print(f"Label (ASCII value): {dataset.labels[0].item()}")
    print(f"Label as character: {chr(dataset.labels[0].item())}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Min label: {metadata['min_label']}, Max label: {metadata['max_label']}")
    print("-" * 50)
    
    print("Dataset processing completed successfully!")

if __name__ == "__main__":
    main()
