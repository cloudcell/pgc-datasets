import os
import sys
import pickle
import argparse
from tqdm import tqdm
import torch
import numpy as np
from collections import Counter

# Add parent directory to path to import pgc_data_lib
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from pgc_data_lib.metadata import create_metadata, validate_classification_labels, save_dataset_with_metadata

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process USPTO chemistry data with unigram or bigram encoding')
parser.add_argument('--mode', type=str, choices=['unigram', 'bigram'], required=True,
                    help='Mode for encoding: unigram (single characters) or bigram (character pairs)')
args = parser.parse_args()

# File paths
# Define the subfolder path
subfolder = os.path.join(os.path.dirname(__file__), "data", "CHEMINFORMATICS")
input_filename = "reactionSmilesFigShareUSPTO2023.txt"
input_filepath = os.path.join(subfolder, input_filename)

# Output filename in the same subfolder
base, ext = os.path.splitext(input_filename)
output_filename = f"{base}_filtered{ext}"
output_filepath = os.path.join(subfolder, output_filename)

max_length = 95

with open(input_filepath, 'r', encoding='utf-8') as infile, open(output_filepath, 'w', encoding='utf-8') as outfile:
    for line in infile:
        formula = line.rstrip('\n')
        if len(formula) <= max_length:
            outfile.write(line)

print(f"Filtering complete. Filtered data saved to: {output_filepath}")

# --- Generate samples from the filtered file (no splitting) ---

def char_to_binary(char):
    """Convert a character to its 8-bit binary representation"""
    ascii_val = ord(char) & 255  # Use full 8 bits
    return np.array([int(b) for b in format(ascii_val, '08b')], dtype=np.uint8)

def generate_samples(input_path, mode='unigram', context_len=98):
    """
    Generate samples from chemistry data with bit encoding and support for unigram/bigram modes
    
    Args:
        input_path: Path to the filtered chemistry data
        mode: 'unigram' or 'bigram' for character or character pair prediction
        context_len: Length of context window in characters
        
    Returns:
        features_tensor: Bit-encoded context features
        labels_tensor: 0-indexed integer labels
        mapping_dict: Dictionary with mappings between indices and characters/bigrams
    """
    print(f"Processing in {mode} mode")
    
    # First pass: collect all unique characters or bigrams for vocabulary
    all_chars = set()
    all_bigrams = set()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Building vocabulary"):
            line = line.rstrip('\n')
            # Find the position after the second '>'
            idx1 = line.find('>')
            idx2 = line.find('>', idx1 + 1) if idx1 != -1 else -1
            if idx2 == -1:
                continue  # Skip lines without two '>'
                
            seq = line[idx2+1:].strip()
            
            # Add all characters to vocabulary
            for ch in seq:
                all_chars.add(ch)
            
            # Add all bigrams to vocabulary
            if mode == 'bigram':
                for i in range(len(seq) - 1):
                    all_bigrams.add(seq[i:i+2])
                # Add the last character with a null char as a special case
                if seq:
                    all_bigrams.add(seq[-1] + '\0')
    
    # Add special characters
    all_chars.add('\0')  # null character for padding
    all_chars.add(';')   # end of formula marker
    
    # Create mappings
    if mode == 'unigram':
        # Unigram mappings
        char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        mapping_dict = {'char_to_idx': char_to_idx, 'idx_to_char': idx_to_char}
        print(f"Unigram vocabulary size: {len(char_to_idx)}")
    else:
        # Bigram mappings
        bigram_to_idx = {bigram: idx for idx, bigram in enumerate(sorted(all_bigrams))}
        idx_to_bigram = {idx: bigram for bigram, idx in bigram_to_idx.items()}
        mapping_dict = {'bigram_to_idx': bigram_to_idx, 'idx_to_bigram': idx_to_bigram}
        print(f"Bigram vocabulary size: {len(bigram_to_idx)}")
    
    # Second pass: generate features and labels
    features = []
    labels = []
    
    # Count total lines for progress bar
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Generating samples"):
            line = line.rstrip('\n')
            # Find the position after the second '>'
            idx1 = line.find('>')
            idx2 = line.find('>', idx1 + 1) if idx1 != -1 else -1
            if idx2 == -1:
                continue  # Skip lines without two '>'
                
            stub = line[:idx2+1]
            seq = line[idx2+1:].strip()
            
            # Pad with null characters instead of spaces
            stub_padded = stub.ljust(context_len, '\0')
            if len(stub_padded) > context_len:
                stub_padded = stub_padded[-context_len:]  # Take last context_len chars if too long
                
            curr = stub_padded
            
            if mode == 'unigram':
                # Unigram mode: predict next character
                for i, ch in enumerate(seq):
                    feature = curr
                    label = ch
                    
                    # Convert feature to bit representation
                    feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                    features.append(feature_bits)
                    
                    # Convert label to 0-indexed integer
                    labels.append(char_to_idx[label])
                    
                    # Shift context window
                    curr = curr[1:] + label
                
                # Add end-of-formula sample
                feature = curr
                label = ';'
                feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                features.append(feature_bits)
                labels.append(char_to_idx[label])
            else:
                # Bigram mode: predict next character pair
                for i in range(len(seq) - 1):
                    feature = curr
                    label = seq[i:i+2]  # Two characters
                    
                    # Convert feature to bit representation
                    feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                    features.append(feature_bits)
                    
                    # Convert label to 0-indexed integer
                    labels.append(bigram_to_idx[label])
                    
                    # Shift context window
                    curr = curr[1:] + seq[i]
                
                # Handle the last character + null
                if seq:
                    feature = curr
                    label = seq[-1] + '\0'  # Last char + null
                    
                    feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                    features.append(feature_bits)
                    labels.append(bigram_to_idx[label])
    
    # Convert to numpy arrays and then to torch tensors
    if features:
        features_np = np.stack(features)
        labels_np = np.array(labels, dtype=np.int64)
        features_tensor = torch.from_numpy(features_np).float()
        labels_tensor = torch.from_numpy(labels_np).long()
        
        print(f"Generated {len(features)} samples")
        print(f"Feature shape: {features_tensor.shape}")
        print(f"Labels range: 0 to {labels_tensor.max().item()}")
        
        return features_tensor, labels_tensor, mapping_dict
    else:
        raise ValueError("No samples were generated. Check the input data.")

# Generate samples from the filtered file with the specified mode
features_tensor, labels_tensor, mapping_dict = generate_samples(
    output_filepath, 
    mode=args.mode, 
    context_len=98
)

# Use plain ASCII codes without remapping
print(f"Using plain ASCII codes: labels range from {labels_tensor.min().item()} to {labels_tensor.max().item()}")

# Create metadata
metadata = create_metadata(
    features=features_tensor,
    labels=labels_tensor,
    dataset_name=f"uspto-small-{args.mode}",
    task_type="classification",
    feature_dim=(features_tensor.shape[1],)
)

# Skip validation since we're using raw ASCII codes that aren't 0-indexed
# validate_classification_labels(metadata)

# No validation since we're using raw ASCII codes

# Define output filename in current directory
output_pkl = f"{base}_filtered_{args.mode}_dataset.pkl"

# Save dataset with metadata
save_dataset_with_metadata(output_pkl, features_tensor, labels_tensor, metadata)

print(f"Processing complete. All data merged into a single dataset.")
print(f"Mode: {args.mode}")
print(f"Total samples: {features_tensor.shape[0]}")
print(f"Feature dimension: {features_tensor.shape[1]}")
print(f"Number of classes: {metadata['num_classes']}")
print(f"Min label: {metadata['min_label']}, Max label: {metadata['max_label']}")
print(f"Dataset saved to: {output_pkl}")

# Print vocabulary size
if args.mode == 'unigram':
    print(f"Vocabulary size (unique characters): {len(mapping_dict['char_to_idx'])}")
    print(f"Sample mappings:")
    for i, (idx, char) in enumerate(list(mapping_dict['idx_to_char'].items())[:5]):
        char_display = repr(char)[1:-1] if char in ['\0', '\n', '\t'] else char
        print(f"  {idx} -> '{char_display}'")
    print("  ...")
else:  # bigram
    print(f"Vocabulary size (unique bigrams): {len(mapping_dict['bigram_to_idx'])}")
    print(f"Sample mappings:")
    for i, (idx, bigram) in enumerate(list(mapping_dict['idx_to_bigram'].items())[:5]):
        bigram_display = ''.join([repr(c)[1:-1] if c in ['\0', '\n', '\t'] else c for c in bigram])
        print(f"  {idx} -> '{bigram_display}'")
    print("  ...")


