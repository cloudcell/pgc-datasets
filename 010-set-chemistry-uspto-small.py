import os
import sys
import pickle
import argparse
import random
import datetime
from tqdm import tqdm
import torch
import numpy as np
from pgc_data_lib.chem2augmented import generate_random_reaction_smiles
from pgc_data_lib.utils import char_to_binary, generate_samples

# Define maximum formula length for context padding
MAX_FORMULA_LENGTH = 64  # 98  # same as context length
MAX_FORMULA_LENGTH_FOR_PREFILTERING = MAX_FORMULA_LENGTH - 3   # 3 characters buffer, just in case

# Add parent directory to path to import pgc_data_lib
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from pgc_data_lib.metadata import create_metadata, validate_classification_labels, save_dataset_with_metadata_h5

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process USPTO chemistry data with unigram, bigram, or trigram encoding')
parser.add_argument('--input', required=True, help='Path to the input file containing reaction SMILES')
parser.add_argument('--mode', required=True, choices=['unigram','bigram','trigram'],
                    help="Mode for encoding: unigram (single characters), bigram (character pairs), or trigram (triplets)")
parser.add_argument('--test_size', type=float, default=0.1, 
                    help='Fraction of data to use for testing (default: 0.1)')
parser.add_argument('--seed', type=int, default=42, 
                    help='Random seed for reproducible train/test splitting (default: 42)')
# augment reaction smiles by specified number of attempts
parser.add_argument('--augment', type=int, default=0, 
                    help='Number of attempts to augment reaction smiles by randomising their representation')
                    
args = parser.parse_args()

# Set random seed for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# File paths
# Get input file path from arguments
input_filepath = os.path.abspath(args.input)
if not os.path.exists(input_filepath):
    raise FileNotFoundError(f"Input file not found: {input_filepath}")

# Generate timestamp for output files
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Define output file paths
output_dir = os.path.dirname(input_filepath)
base_filename = os.path.splitext(os.path.basename(input_filepath))[0]
output_filepath = os.path.join(output_dir, f"{base_filename}_filtered_{timestamp}.txt")

# Filter the input file to remove lines without two '>' and formulas longer than MAX_FORMULA_LENGTH_FOR_PREFILTERING
print(f"Filtering data from {input_filepath}...")
total_lines = 0
filtered_lines = 0
filtered_data = []

with open(input_filepath, 'r') as f_in:
    for line in f_in:
        total_lines += 1
        # Check for required '>' characters
        if line.count('>') >= 2:
            # Check entire formula length (the whole line)
            if len(line) <= MAX_FORMULA_LENGTH_FOR_PREFILTERING:
                filtered_data.append(line)
                filtered_lines += 1

print(f"Filtering complete. Kept {filtered_lines} out of {total_lines} lines.")

# Split the filtered data into train/val and test sets
random.shuffle(filtered_data)  # Shuffle the data before splitting
test_size = int(len(filtered_data) * args.test_size)
train_data = filtered_data[test_size:]
test_data = filtered_data[:test_size]

print(f"Split data into {len(train_data)} training samples and {len(test_data)} testing samples.")

# Save the split datasets to files
train_filepath = os.path.join(output_dir, f"{base_filename}_train_{timestamp}.txt")
test_filepath = os.path.join(output_dir, f"{base_filename}_test_{timestamp}.txt")

with open(train_filepath, 'w') as f_out:
    f_out.writelines(train_data)

with open(test_filepath, 'w') as f_out:
    f_out.writelines(test_data)

print(f"Training data saved to: {train_filepath}")
print(f"Testing data saved to: {test_filepath}")

# --- Generate samples from the split files ---

# Process training data
print("\nProcessing training data...")
train_features_tensor, train_labels_tensor, train_label_chars = generate_samples(
    train_filepath,
    mode=args.mode,
    augment_nbr=args.augment,
    context_len=MAX_FORMULA_LENGTH
)

# Process testing data
print("\nProcessing testing data...")
test_features_tensor, test_labels_tensor, test_label_chars = generate_samples(
    test_filepath,
    mode=args.mode,
    augment_nbr=args.augment,
    context_len=MAX_FORMULA_LENGTH
)

# Create metadata for both datasets
if args.mode == 'unigram':
    num_classes = 256
elif args.mode == 'bigram':
    num_classes = 256 ** 2
elif args.mode == 'trigram':
    num_classes = 256 ** 3
else:
    num_classes = None

# Create metadata for training dataset
train_metadata = create_metadata(
    features=train_features_tensor,
    labels=train_labels_tensor,
    dataset_name=f"chem-uspto2023-small-{args.mode}-train",
    task_type="classification",
    feature_dim=(train_features_tensor.shape[1],),
    num_classes=num_classes
)

# Create metadata for testing dataset
test_metadata = create_metadata(
    features=test_features_tensor,
    labels=test_labels_tensor,
    dataset_name=f"chem-uspto2023-small-{args.mode}-test",
    task_type="classification",
    feature_dim=(test_features_tensor.shape[1],),
    num_classes=num_classes
)

# Validate the metadata
validate_classification_labels(train_metadata)
validate_classification_labels(test_metadata)

# Save the datasets with metadata
train_h5 = os.path.join(output_dir, f"dataset_010-chemistry_uspto_{args.mode}_train_{timestamp}.h5")
test_h5 = os.path.join(output_dir, f"dataset_010-chemistry_uspto_{args.mode}_test_{timestamp}.h5")

save_dataset_with_metadata_h5(
    train_h5,
    train_features_tensor,
    train_labels_tensor,
    train_metadata
)

save_dataset_with_metadata_h5(
    test_h5,
    test_features_tensor,
    test_labels_tensor,
    test_metadata
)

print(f"\nTraining dataset saved to {train_h5}")
print(f"Testing dataset saved to {test_h5}")

# # Save label characters for reference (optional)
# train_labels_pkl = os.path.join(output_dir, f"chemistry_uspto_{args.mode}_train_labels_{timestamp}.pkl")
# test_labels_pkl = os.path.join(output_dir, f"chemistry_uspto_{args.mode}_test_labels_{timestamp}.pkl")

# with open(train_labels_pkl, 'wb') as f:
#     pickle.dump(train_label_chars, f)

# with open(test_labels_pkl, 'wb') as f:
#     pickle.dump(test_label_chars, f)

print("\nProcessing complete.")
print(f"Random seed used: {args.seed}")
print(f"Test size: {args.test_size} ({len(test_data)} samples)")
print(f"Training size: {1-args.test_size} ({len(train_data)} samples)")
print(f"Feature dimension: {train_features_tensor.shape[1]}")
print(f"Number of classes: {train_metadata['num_classes']}")
print(f"Min label: {train_metadata['min_label']}, Max label: {train_metadata['max_label']}")
print(f"Done.")
