"""
Utility functions for PGC datasets.
"""
import numpy as np
from tqdm import tqdm
import torch

def char_to_binary(char):
    """Convert a character to its 8-bit binary representation (0-255)."""
    ascii_val = ord(char) & 255  # Use full 8 bits
    return np.array([int(b) for b in format(ascii_val, '08b')], dtype=np.uint8)


def generate_samples(input_path, mode='unigram', context_len=98):
    """
    Generate samples from text data with bit encoding and support for unigram/bigram/trigram modes
    
    Args:
        input_path: Path to the input data file
        mode: 'unigram', 'bigram', or 'trigram' for character prediction mode
        context_len: Length of context window in characters
        
    Returns:
        features_tensor: Bit-encoded context features as numpy array
        labels_tensor: Raw integer labels (ASCII values or combined ASCII values) as numpy array
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
            seq = line.rstrip('\n')
            if len(seq) < context_len + 1:
                continue
                
            curr = seq[:context_len]
            
            if mode == 'unigram':
                # Unigram mode: predict next single character
                for i in range(context_len, len(seq)):
                    feature = curr
                    label = seq[i]
                    
                    # Convert feature to bit representation
                    feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                    features.append(feature_bits)
                    labels.append(ord(label))
                    label_chars.append(label)
                    
                    # Shift context window
                    curr = curr[1:] + label
                    
            elif mode == 'bigram':
                # Bigram mode: predict next character pair using combined ASCII values
                for i in range(context_len, len(seq) - 1):
                    feature = curr
                    bigram = seq[i:i+2]  # Two characters
                    
                    # Convert feature to bit representation
                    feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                    features.append(feature_bits)
                    
                    # Store combined ASCII value as label (first char << 8 + second char)
                    label_val = (ord(bigram[0]) << 8) | ord(bigram[1])
                    labels.append(label_val)
                    label_chars.append(bigram)
                    
                    # Shift context window
                    curr = curr[1:] + seq[i]
                    
            elif mode == 'trigram':
                # Trigram mode: predict next character triplet using combined ASCII values
                for i in range(context_len, len(seq) - 2):
                    feature = curr
                    trigram = seq[i:i+3]
                    
                    # Convert feature to bit representation
                    feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                    features.append(feature_bits)
                    
                    # Store combined ASCII value as label
                    label_val = (ord(trigram[0]) << 16) | (ord(trigram[1]) << 8) | ord(trigram[2])
                    labels.append(label_val)
                    label_chars.append(trigram)
                    
                    # Shift context window
                    curr = curr[1:] + seq[i]
            else:
                raise ValueError("mode must be 'unigram', 'bigram', or 'trigram'")
                
    if features:
        features_tensor = np.stack(features)
        labels_tensor = np.array(labels)
        
        print(f"Feature shape: {features_tensor.shape}")
        print(f"Labels range: {labels_tensor.min()} to {labels_tensor.max()}")
        print(f"Using plain ASCII codes without remapping")
        
        return features_tensor, labels_tensor, label_chars
    else:
        raise ValueError("No samples were generated. Check the input data.")
