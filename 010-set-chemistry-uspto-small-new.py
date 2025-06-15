import os
import sys
import pickle
import argparse
from tqdm import tqdm
import torch
import numpy as np

# Add parent directory to path to import pgc_data_lib
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from pgc_data_lib.metadata import create_metadata, validate_classification_labels, save_dataset_with_metadata

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process USPTO chemistry data with unigram or bigram encoding')
parser.add_argument('--mode', type=str, choices=['unigram', 'bigram'], required=True,
                    help='Mode for encoding: unigram (single characters) or bigram (character pairs)')
args = parser.parse_args()

# File paths
# Define the subfolder path for input only
subfolder = os.path.join(os.path.dirname(__file__), "data", "CHEMINFORMATICS")
os.makedirs(subfolder, exist_ok=True)

# Define input file path (in subfolder)
input_filepath = os.path.join(subfolder, "reactionSmilesFigShareUSPTO2023.txt")
base, ext = os.path.splitext(os.path.basename(input_filepath))

# Define output file paths (in current directory)
output_filepath = f"{base}_filtered{ext}"

# Filter the input file to remove lines without two '>'
if not os.path.exists(output_filepath):
    with open(input_filepath, 'r', encoding='utf-8') as f_in, open(output_filepath, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if line.count('>') >= 2:
                f_out.write(line)

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
        labels_tensor: Raw integer labels (ASCII values or combined ASCII values)
        label_chars: Original character representations of the labels
    """
    print(f"Processing in {mode} mode")
    
    # Generate features and labels
    features = []
    labels = []
    label_chars = []  # Store original character representations
    
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
                # Unigram mode: predict next character using ASCII values
                for i, ch in enumerate(seq):
                    feature = curr
                    
                    # Convert feature to bit representation
                    feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                    features.append(feature_bits)
                    
                    # Store raw ASCII value as label
                    labels.append(ord(ch))
                    label_chars.append(ch)
                    
                    # Shift context window
                    curr = curr[1:] + ch
                
                # Add end-of-formula sample
                feature = curr
                label = ';'
                feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                features.append(feature_bits)
                labels.append(ord(label))
                label_chars.append(label)
            else:
                # Bigram mode: predict next character pair using combined ASCII values
                for i in range(len(seq) - 1):
                    feature = curr
                    bigram = seq[i:i+2]  # Two characters
                    
                    # Convert feature to bit representation
                    feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                    features.append(feature_bits)
                    
                    # Store combined ASCII value as label (first char * 256 + second char)
                    label_val = ord(bigram[0]) * 256 + ord(bigram[1])
                    labels.append(label_val)
                    label_chars.append(bigram)
                    
                    # Shift context window
                    curr = curr[1:] + seq[i]
                
                # Handle the last character + null
                if seq:
                    feature = curr
                    bigram = seq[-1] + '\0'  # Last char + null
                    
                    feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                    features.append(feature_bits)
                    
                    label_val = ord(bigram[0]) * 256 + ord(bigram[1])
                    labels.append(label_val)
                    label_chars.append(bigram)
    
    # Convert to numpy arrays and then to torch tensors
    if features:
        features_np = np.stack(features)
        labels_np = np.array(labels)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        print(f"Feature shape: {features_tensor.shape}")
        print(f"Labels range: {labels_tensor.min().item()} to {labels_tensor.max().item()}")
        print(f"Using plain ASCII codes without remapping")
        
        return features_tensor, labels_tensor, label_chars
    else:
        raise ValueError("No samples were generated. Check the input data.")

# Generate samples from the filtered file with the specified mode
features_tensor, raw_labels_tensor, label_chars = generate_samples(
    output_filepath, 
    mode=args.mode, 
    context_len=98
)

# Create metadata using the library function
metadata = create_metadata(
    features=features_tensor,
    labels=raw_labels_tensor,
    dataset_name=f"uspto-small-{args.mode}",
    task_type="classification",
    feature_dim=(features_tensor.shape[1],)
)

# Define output filename in current directory
output_pkl = f"{base}_filtered_{args.mode}_dataset.pkl"

# Save dataset with metadata
save_dataset_with_metadata(output_pkl, features_tensor, raw_labels_tensor, metadata)

print(f"Processing complete. All data merged into a single dataset.")
print(f"Mode: {args.mode}")
print(f"Total samples: {features_tensor.shape[0]}")
print(f"Feature dimension: {features_tensor.shape[1]}")
print(f"Number of classes: {metadata['num_classes']}")
print(f"Min label: {metadata['min_label']}, Max label: {metadata['max_label']}")
print(f"Dataset saved to: {output_pkl}")
