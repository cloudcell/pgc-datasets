# Code for Paper: "Polymorphic Graph Classifier"
# http://dx.doi.org/10.13140/RG.2.2.15744.55041
# Design: Alexander Bikeyev
# Date: 2025-04-20
# LICENSE: AGPL v3


CONTEXT_SIZE = 98  # character context size


import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm
import sys
import argparse

class TextBinaryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def char_to_binary(char):
    """Convert a character to its 8-bit binary representation (0-255)."""
    val = ord(char)
    return [int(b) for b in format(val, '08b')]

def prepend_zeros_to_text(text, context_size=CONTEXT_SIZE):
    """
    Prepend the text with as many zero ASCII characters (\x00) as context_size.
    Args:
        text (bytes or str): The input text or bytes.
        context_size (int): Number of zero bytes/chars to prepend. Defaults to CONTEXT_SIZE.
    Returns:
        Same type as input: The text/bytes with prepended zero bytes/chars.
    """
    if isinstance(text, bytes):
        return (b'\x00' * context_size) + text
    elif isinstance(text, str):
        return ('\x00' * context_size) + text
    else:
        raise TypeError(f"Unsupported type for prepend_zeros_to_text: {type(text)}")


def process_text_file(file_path, chunk_type):
    """Process text file and convert to binary sequences with sliding windows.
    chunk_type: 'unigram' or 'bigram'
    """
    with open(file_path, 'rb') as f:
        data = f.read()

    features = []
    labels = []
    window_size = CONTEXT_SIZE * 8  # Same as MNIST

    data = prepend_zeros_to_text(data, context_size=CONTEXT_SIZE)

    if chunk_type == 'unigram':
        # Process as single bytes
        binary_data = []
        byte_list = []
        print("Converting bytes to binary (unigram mode)...")
        for b in tqdm(data, desc="Processing bytes"):
            binary_data.extend([int(bit) for bit in format(b, '08b')])
            byte_list.append(b)
        # Window slides by 1 byte (8 bits)
        if len(binary_data) >= window_size + 8:
            total_windows = len(binary_data) - window_size - 8 + 1
            print("\nCreating sliding windows...")
            for i in tqdm(range(0, total_windows, 8), desc="Creating samples"):
                window_start = i
                window = binary_data[window_start:window_start + window_size]
                next_byte_pos = (window_start + window_size) // 8
                if next_byte_pos < len(byte_list):
                    features.append(window)
                    labels.append(byte_list[next_byte_pos] & 127)
        return features, labels

    elif chunk_type == 'bigram':
        # Prepare binary data for all bytes
        binary_data = []
        for b in data:
            binary_data.extend([int(bit) for bit in format(b, '08b')])
        # Window slides by 1 byte (8 bits)
        if len(binary_data) >= window_size + 16:
            total_windows = len(binary_data) - window_size - 8 + 1  # slide by 8 bits
            print("\nCreating sliding windows...")
            for i in tqdm(range(0, total_windows, 8), desc="Creating samples"):
                window_start = i
                window = binary_data[window_start:window_start + window_size]
                next_byte_pos = (window_start + window_size) // 8
                # Prepare label as the next bigram (pad with 0 if needed)
                if next_byte_pos < len(data):
                    b1 = data[next_byte_pos]
                    b2 = data[next_byte_pos+1] if (next_byte_pos+1) < len(data) else 0
                    label = ((b1 & 0xFF) << 8) | (b2 & 0xFF)
                    features.append(window)
                    labels.append(label)
        return features, labels
    elif chunk_type == 'trigram':
        binary_data = []
        for b in data:
            binary_data.extend([int(bit) for bit in format(b, '08b')])
        # Window slides by 1 byte (8 bits)
        if len(binary_data) >= window_size + 24:
            total_windows = len(binary_data) - window_size - 8 + 1  # slide by 8 bits
            print("\nCreating sliding windows...")
            for i in tqdm(range(0, total_windows, 8), desc="Creating samples"):
                window_start = i
                window = binary_data[window_start:window_start + window_size]
                next_byte_pos = (window_start + window_size) // 8
                # Prepare label as the next trigram (pad with 0 if needed)
                if next_byte_pos < len(data):
                    b1 = data[next_byte_pos]
                    b2 = data[next_byte_pos+1] if (next_byte_pos+1) < len(data) else 0
                    b3 = data[next_byte_pos+2] if (next_byte_pos+2) < len(data) else 0
                    label = ((b1 & 0xFF) << 16) | ((b2 & 0xFF) << 8) | (b3 & 0xFF)
                    features.append(window)
                    labels.append(label)
        return features, labels
    else:
        raise ValueError("chunk_type must be 'unigram', 'bigram', or 'trigram'")

def main():
    parser = argparse.ArgumentParser(description="Convert text file to binary dataset with sliding window.")
    parser.add_argument('input_file', help='Input text file')
    parser.add_argument('output_file', help='Output dataset file')
    parser.add_argument('--chunk', required=True, choices=['unigram', 'bigram', 'trigram'], help="Chunking method: 'unigram' (single char), 'bigram' (pairs), or 'trigram' (triplets, pad with ASCII 0 if needed)")
    args = parser.parse_args()

    # Process the text file
    features, labels = process_text_file(args.input_file, args.chunk)

    # Create the dataset
    dataset = TextBinaryDataset(features, labels)

    # Save the dataset
    with open(args.output_file, 'wb') as f:
        pickle.dump({'features': dataset.features, 'labels': dataset.labels}, f)

    print(f"Dataset created with {len(dataset)} samples")
    print(f"Saved dataset to {args.output_file}")

if __name__ == "__main__":
    main()
