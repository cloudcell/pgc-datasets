import os
import sys
import pickle
import argparse
from tqdm import tqdm
import torch
import numpy as np
from pgc_data_lib.chem2augmented import generate_random_reaction_smiles

# Define maximum formula length for context padding
MAX_FORMULA_LENGTH = 64  # 98  # same as context length
MAX_FORMULA_LENGTH_FOR_PREFILTERING = MAX_FORMULA_LENGTH - 3   # 3 characters buffer, just in case

# Add parent directory to path to import pgc_data_lib
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from pgc_data_lib.metadata import create_metadata, validate_classification_labels, save_dataset_with_metadata_h5
from pgc_data_lib.utils import char_to_binary, generate_samples

# Parse command line arguments
parser = argparse.ArgumentParser(description='Process USPTO chemistry data with unigram, bigram, or trigram encoding')
parser.add_argument('--mode', required=True, choices=['unigram','bigram','trigram'],
                    help="Mode for encoding: unigram (single characters), bigram (character pairs), or trigram (triplets)")
# augment reaction smiles by specified number of attempts
parser.add_argument('--augment', type=int, default=0, help='Number of attempts to augment reaction smiles by randomising their representation')
                    
args = parser.parse_args()

# File paths
# Define the subfolder path for input only
subfolder = os.path.join(os.path.dirname(__file__), "data", "CHEMINFORMATICS")
os.makedirs(subfolder, exist_ok=True)

# Define input file path (in subfolder)
input_filepath = os.path.join(subfolder, "reactionSmilesFigShareUSPTO2023.txt")

# Define output file paths (in current directory)
output_filepath = f"{input_filepath}_filtered.txt"

# Filter the input file to remove lines without two '>' and formulas longer than MAX_FORMULA_LENGTH_FOR_PREFILTERING
if not os.path.exists(output_filepath):
    print(f"Filtering data...")
    total_lines = 0
    filtered_lines = 0
    with open(input_filepath, 'r') as f_in, open(output_filepath, 'w') as f_out:
        for line in f_in:
            total_lines += 1
            # Check for required '>' characters
            if line.count('>') >= 2:
                # Check entire formula length (the whole line)
                if len(line) <= MAX_FORMULA_LENGTH_FOR_PREFILTERING:
                    f_out.write(line)
                    filtered_lines += 1
    print(f"Filtering complete. Kept {filtered_lines} out of {total_lines} lines.")
    print(f"Filtered data saved to: {output_filepath}")
else:
    print(f"Using existing filtered data from: {output_filepath}")

# --- Generate samples from the filtered file (no splitting) ---

# Generate samples from the filtered file with the specified mode
features_tensor, labels_tensor, label_chars = generate_samples(
    output_filepath,
    mode=args.mode,
    context_len=MAX_FORMULA_LENGTH
)

# Create metadata using the library function
if args.mode == 'unigram':
    num_classes = 256
elif args.mode == 'bigram':
    num_classes = 256 ** 2
elif args.mode == 'trigram':
    num_classes = 256 ** 3
else:
    num_classes = None
metadata = create_metadata(
    features=features_tensor,
    labels=labels_tensor,
    dataset_name=f"chem-uspto2023-small-{args.mode}",
    task_type="classification",
    feature_dim=(features_tensor.shape[1],),
    num_classes=num_classes
)

# Define output filename in current directory
output_pkl = f"010-chem-uspto2023-small-{args.mode}"

# Save dataset with metadata
save_dataset_with_metadata_h5(output_pkl, features_tensor, labels_tensor, metadata)

print(f"Processing complete. All data merged into a single dataset.")
print(f"Mode: {args.mode}")
print(f"Total samples: {features_tensor.shape[0]}")
print(f"Feature dimension: {features_tensor.shape[1]}")
print(f"Number of classes: {metadata['num_classes']}")
print(f"Min label: {metadata['min_label']}, Max label: {metadata['max_label']}")
print(f"Done.")
