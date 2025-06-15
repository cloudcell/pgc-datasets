# Code for Paper: "Polymorphic Graph Classifier"
# http://dx.doi.org/10.13140/RG.2.2.15744.55041
# Design: Alexander Bikeyev
# Date: 2025-04-20
# LICENSE: AGPL v3

"""
This script converts a text file to a binary dataset with sliding windows.
It supports unigram, bigram, and trigram encoding.

Usage:
    python txt2bin.py <input_file> <output_file> --num_features <num_features> --class_type <class_type> [--no-prepend]

    input_file: Path to the text file to process
    output_file: Path to the output binary file
    num_features: Number of binary features per sample (must be divisible by 8)
    class_type: 'unigram', 'bigram', or 'trigram'
    --no-prepend: Do not prepend null characters (default: prepend)
    --no-append: Do not append null characters (default: append)

"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm
import sys
import argparse
import tempfile
from pgc_data_lib.metadata import create_metadata, save_dataset_with_metadata

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

def process_text_file(file_path, num_features, feature_type, prepend_nulls=True, append_nulls=True):
    """Process text file and convert to binary sequences with sliding windows.
    Args:
        file_path: Path to the text file to process
        num_features: Number of binary features per sample (must be divisible by 8)
        feature_type: 'unigram', 'bigram', or 'trigram'
        prepend_nulls: Whether to prepend null characters the size of num_features/8
        append_nulls: Whether to append null characters the size of num_features/8
    """
    # use binary encoding
    with open(file_path, 'rb') as f:
        text = f.read()

    binary_data = []
    ascii_chars = []
    window_size = num_features
    if feature_type == 'unigram':
        char_group = 1
    elif feature_type == 'bigram':
        char_group = 2
    elif feature_type == 'trigram':
        char_group = 3
    else:
        raise ValueError("feature_type must be 'unigram', 'bigram', or 'trigram'")
    
    # Prepend null chars if needed at the beginning of the text
    if prepend_nulls:
        null_chars_count = window_size // 8
        print(f"Prepending {null_chars_count} null characters...")
        for _ in range(null_chars_count):
            binary_data.extend(char_to_binary('\0'))
            ascii_chars.append('\0')


    print("Converting characters to binary...")
    for char in tqdm(text, desc="Processing characters"):
        c = chr(char)
        binary_data.extend(char_to_binary(c))
        ascii_chars.append(c)



    # append null characters to the end of the text depending on feature_type
    if append_nulls:
        if feature_type == 'unigram':
            binary_data.extend(char_to_binary('\0'))
            ascii_chars.append('\0')  # a file must end with null character
        elif feature_type == 'bigram':
            for c in '\0\0':
                binary_data.extend(char_to_binary(c))
            ascii_chars.append('\0')
            ascii_chars.append('\0')  # append two null characters
        elif feature_type == 'trigram':
            for c in '\0\0\0':
                binary_data.extend(char_to_binary(c))
            ascii_chars.append('\0')
            ascii_chars.append('\0')
            ascii_chars.append('\0')  # append three null characters
    

    features = []
    labels = []
    # Always slide by 1 character (8 bits)
    if len(binary_data) >= window_size + 8:
        total_windows = (len(binary_data) - window_size) // 8
        print("\nCreating sliding windows...")
        for i in tqdm(range(total_windows), desc="Creating samples"):
            window_start = i * 8
            window = binary_data[window_start:window_start + window_size]
            next_char_idx = (window_start + window_size) // 8
            # Label: next char(s) after the window
            if feature_type == 'unigram':
                if next_char_idx < len(ascii_chars):
                    features.append(window)
                    labels.append(ord(ascii_chars[next_char_idx]))
            elif feature_type == 'bigram':
                if next_char_idx + 1 < len(ascii_chars):
                    b1 = ord(ascii_chars[next_char_idx])
                    b2 = ord(ascii_chars[next_char_idx+1])
                    label = (b1 << 8) | b2
                    features.append(window)
                    labels.append(label)
            elif feature_type == 'trigram':
                if next_char_idx + 2 < len(ascii_chars):
                    b1 = ord(ascii_chars[next_char_idx])
                    b2 = ord(ascii_chars[next_char_idx+1])
                    b3 = ord(ascii_chars[next_char_idx+2])
                    label = (b1 << 16) | (b2 << 8) | b3
                    features.append(window)
                    labels.append(label)
            else:
                raise ValueError("feature_type must be 'unigram', 'bigram', or 'trigram'")
    return features, labels

def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="Convert text to binary dataset")
        parser.add_argument('input_file', help='Input file. Use \'-\' to read from stdin')
        parser.add_argument('output_file', help='Output file. Use \'-\' to write to stdout')
        parser.add_argument('--num_features', type=int, required=True, help='Number of binary features (must be divisible by 8, e.g. 784)')
        parser.add_argument('--class_type', required=True, choices=['unigram', 'bigram', 'trigram'], help="Class type: 'unigram', 'bigram', or 'trigram'")
        parser.add_argument('--no-prepend', action='store_true', help='Do not prepend null characters (default: prepend)')
        parser.add_argument('--no-append', action='store_true', help='Do not append null characters (default: append)')
        args = parser.parse_args()

        # Validate num_features
        if args.num_features % 8 != 0:
            print(f"Error: --num_features must be divisible by 8 (got {args.num_features}). To read N chars, set --num_features to N*8. For example, to use 98 chars, use --num_features 784.")
            sys.exit(2)
        num_chars = args.num_features // 8
        print(f"Using num_features={args.num_features} (i.e. {num_chars} chars per sample)")

        # Process input
        if args.input_file == '-':
            content = sys.stdin.buffer.read()
            if not content:
                print("Error: No input received from stdin", file=sys.stderr)
                sys.exit(1)
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                input_file = temp_file.name
        else:
            input_file = args.input_file

        # Process the text file
        features, labels = process_text_file(input_file, args.num_features, args.class_type, prepend_nulls=not args.no_prepend, append_nulls=not args.no_append)

        # Create the dataset
        dataset = TextBinaryDataset(features, labels)

        # Prepare metadata using the library
        dataset_name = f"txt2bin-{args.class_type}-{args.num_features}"
        if args.class_type == 'unigram':
            num_classes = 256
        elif args.class_type == 'bigram':
            num_classes = 256 ** 2
        elif args.class_type == 'trigram':
            num_classes = 256 ** 3
        else:
            num_classes = None
        metadata = create_metadata(dataset.features, dataset.labels, dataset_name=dataset_name, task_type="classification", feature_dim=(args.num_features,), num_classes=num_classes)

        # Save dataset with metadata using the library
        save_dataset_with_metadata(args.output_file, dataset.features, dataset.labels, metadata)

        print(f"Dataset created with {len(dataset)} samples")

        # Clean up temporary file if it was created
        if args.input_file == '-':
            os.unlink(input_file)

    except KeyboardInterrupt:
        print("\nOperation interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error processing dataset: {str(e)}", file=sys.stderr)
        if 'input_file' in locals() and args.input_file == '-':
            try:
                os.unlink(input_file)
            except:
                pass
        sys.exit(1)

if __name__ == "__main__":
    main()
