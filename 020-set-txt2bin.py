# Code for Paper: "Polymorphic Graph Classifier"
# http://dx.doi.org/10.13140/RG.2.2.15744.55041
# Design: Alexander Bikeyev
# Date: 2025-04-20
# LICENSE: AGPL v3


import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from tqdm import tqdm
import sys
import argparse
import tempfile

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

def process_text_file(file_path, prepend_nulls=True):
    """Process text file and convert to binary sequences with sliding windows.
    
    Args:
        file_path: Path to the text file to process
        prepend_nulls: Whether to prepend null characters the size of num_features
    """
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Convert all valid ASCII characters to binary
    binary_data = []
    ascii_chars = []
    
    # If prepend_nulls is True, prepend null characters the size of window_size/8
    # This helps with initial prediction where there's no context
    window_size = 784  # Same as MNIST
    if prepend_nulls:
        null_chars_count = window_size // 8  # Number of null chars needed to fill one window
        print(f"Prepending {null_chars_count} null characters...")
        # Prepend null characters
        for _ in range(null_chars_count):
            binary_data.extend(char_to_binary('\0'))
            ascii_chars.append('\0')
    
    print("Converting characters to binary...")
    for char in tqdm(text, desc="Processing characters"):
        binary_data.extend(char_to_binary(char))
        ascii_chars.append(char)
    
    # Create samples using sliding window
    features = []
    labels = []
    
    # We need at least window_size binary digits plus one character for the label
    if len(binary_data) >= window_size + 8:
        # Calculate number of possible windows when sliding by 8 bits
        total_windows = (len(binary_data) - window_size) // 8
        print("\nCreating sliding windows...")
        for i in tqdm(range(total_windows), desc="Creating samples"):
            # Get window starting at i*8 (sliding by 8 bits each time)
            window_start = i * 8
            window = binary_data[window_start:window_start + window_size]
            
            # The next character after our window
            # Calculate the character position by dividing by 8 (bits per char)
            # Since window_start is in bits, we need to add window_size and then
            # find which character position this corresponds to
            next_char_idx = (window_start + window_size) // 8
            if next_char_idx < len(ascii_chars):
                features.append(window)
                # Use the full ASCII value without masking to preserve all information
                labels.append(ord(ascii_chars[next_char_idx]))
    
    return features, labels

def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description="Convert text to binary dataset")
        parser.add_argument('input_file', help='Input file. Use \'-\' to read from stdin')
        parser.add_argument('output_file', help='Output file. Use \'-\' to write to stdout')
        parser.add_argument('--no-prepend', action='store_true', help='Do not prepend null characters (default: prepend)')
        args = parser.parse_args()

        # Process input
        if args.input_file == '-':
            # Read from stdin
            content = sys.stdin.buffer.read()
            if not content:
                print("Error: No input received from stdin", file=sys.stderr)
                sys.exit(1)
            # Create temporary file for input
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(content)
                input_file = temp_file.name
        else:
            input_file = args.input_file

        # Process the text file
        features, labels = process_text_file(input_file, prepend_nulls=not args.no_prepend)

        # Create the dataset
        dataset = TextBinaryDataset(features, labels)

        # Handle output
        if args.output_file == '-':
            # Write to stdout
            try:
                sys.stdout.buffer.write(pickle.dumps({
                    'features': dataset.features,
                    'labels': dataset.labels
                }))
                sys.stdout.buffer.flush()
            except BrokenPipeError:
                print("Error: Broken pipe detected", file=sys.stderr)
                sys.exit(1)
        else:
            # Write to file
            with open(args.output_file, 'wb') as f:
                pickle.dump({
                    'features': dataset.features,
                    'labels': dataset.labels
                }, f)

        print(f"Dataset created with {len(dataset)} samples")
        if args.output_file == '-':
            print("Dataset written to stdout")
        else:
            print(f"Saved dataset to {args.output_file}")

        # Clean up temporary file if it was created
        if args.input_file == '-':
            os.unlink(input_file)

    except KeyboardInterrupt:
        print("\nOperation interrupted by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Error processing dataset: {str(e)}", file=sys.stderr)
        if args.input_file == '-':
            try:
                os.unlink(input_file)
            except:
                pass
        sys.exit(1)

if __name__ == "__main__":
    main()
