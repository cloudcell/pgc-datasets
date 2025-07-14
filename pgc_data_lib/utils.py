"""
Utility functions for PGC datasets.
"""
import numpy as np
from tqdm import tqdm
import torch
from pgc_data_lib.chem2augmented import generate_random_reaction_smiles
import rdkit.Chem as Chem
from rdkit.Chem import AllChem  # For reaction parsing

def char_to_binary(char):
    """Convert a character to its 8-bit binary representation (0-255)."""
    ascii_val = ord(char) & 255  # Use full 8 bits
    return np.array([int(b) for b in format(ascii_val, '08b')], dtype=np.uint8)


def generate_samples(input_path, mode='unigram', augment_nbr=0, context_len=98, debug=True, min_valid_samples=1):
    """
    Generate samples from chemistry data with bit encoding and support for unigram/bigram modes
    
    Args:
        input_path: Path to the filtered chemistry data
        mode: 'unigram' or 'bigram' for character or character pair prediction
        augment_nbr: Number of attempts to augment reaction smiles by randomising their representation
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
    
    # Counters for debugging
    total_smiles = 0
    valid_smiles = 0
    invalid_smiles = 0
    invalid_format = 0
    missing_separators = 0
    rdkit_parse_errors = 0
    other_errors = 0
    
    # Sample some problematic SMILES for debugging
    problematic_samples = []
    
    # Count total lines for progress bar
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
        
    with open(input_path, 'r', encoding='utf-8') as f:

        for line in tqdm(f, total=total_lines, desc="Generating samples"):
            total_smiles += 1
            line_raw = line.rstrip('\n')
            
            # Skip empty lines or lines with just separators
            if not line_raw or line_raw.strip() == '>>' or not '>>' in line_raw:
                invalid_smiles += 1
                other_errors += 1
                if len(problematic_samples) < 5:
                    problematic_samples.append(("Empty or malformed reaction", line_raw))
                continue
                
            if augment_nbr > 0:
                try:
                    augmented_smiles_dict = generate_random_reaction_smiles(line_raw, max_attempts=augment_nbr, random_state=42, product_canonical=True)
                    augmented_smiles_list = list(augmented_smiles_dict.values())
                except Exception as e:
                    print(f"WARNING: Error augmenting SMILES: {line_raw}, error: {str(e)}")
                    invalid_smiles += 1
                    other_errors += 1
                    if len(problematic_samples) < 5:
                        problematic_samples.append(("Augmentation error", line_raw))
                    continue
            else:
                # product must be canonical
                try:
                    # Parse as a reaction instead of a molecule
                    rxn = AllChem.ReactionFromSmarts(line_raw, useSmiles=True)
                    if rxn is None:
                        print(f"WARNING: RDKit could not parse reaction SMILES: {line_raw}")
                        invalid_smiles += 1
                        rdkit_parse_errors += 1
                        if len(problematic_samples) < 5:
                            problematic_samples.append(("RDKit parse error", line_raw))
                        continue
                    
                    # Convert back to canonical form if needed
                    if True:  # Always canonicalize
                        try:
                            line_raw = AllChem.ReactionToSmiles(rxn)
                        except Exception as e:
                            print(f"WARNING: Error converting reaction to canonical SMILES: {line_raw}, error: {str(e)}")
                            # Continue with the original SMILES
                    
                    valid_smiles += 1
                    augmented_smiles_list = [line_raw]
                except Exception as e:
                    print(f"WARNING: Unexpected error parsing SMILES: {line_raw}, error: {str(e)}")
                    invalid_smiles += 1
                    other_errors += 1
                    if len(problematic_samples) < 5:
                        problematic_samples.append(("Unexpected error", line_raw))
                    continue

            for i in range(len(augmented_smiles_list)):
                line = augmented_smiles_list[i]

                # Find the position after the second '>'
                idx1 = line.find('>')
                idx2 = line.find('>', idx1 + 1) if idx1 != -1 else -1
                if idx2 == -1:
                    missing_separators += 1
                    if len(problematic_samples) < 5:
                        problematic_samples.append(("Missing separators", line))
                    continue  # Skip lines without two '>'
                    
                stub = line[:idx2+1]
                seq = line[idx2+1:].strip()

                # add end of formula character for unigram, bigram, and trigram
                if mode == 'unigram':
                    seq += ';'
                elif mode == 'bigram':
                    seq += ';\0'
                elif mode == 'trigram':
                    seq += ';\0\0'

                
                # Pad with null characters on the left instead of the right
                stub_padded = stub.rjust(context_len, '\0')
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

                        # if the last character of the curr is end of formula character, then break
                        if curr[-1] == ';':
                            break
                    
                elif mode == 'bigram':
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

                        # if the last character of the curr is end of formula character, then break
                        if curr[-1] == ';':
                            break
                    
                elif mode == 'trigram':
                    # Trigram mode: predict next character triplet using combined ASCII values
                    for i in range(len(seq)):
                        feature = curr
                        # Always get 3 chars, pad with nulls if needed
                        if i+2 < len(seq):
                            trigram = seq[i:i+3]
                        elif i+1 < len(seq):
                            trigram = seq[i:i+2] + '\0'
                        else:
                            trigram = seq[i] + '\0\0'
                        # Convert feature to bit representation
                        feature_bits = np.concatenate([char_to_binary(c) for c in feature])
                        features.append(feature_bits)
                        # Store combined ASCII value as label
                        label_val = (ord(trigram[0]) << 16) + (ord(trigram[1]) << 8) + ord(trigram[2])
                        labels.append(label_val)
                        label_chars.append(trigram)
                        # Shift context window
                        curr = curr[1:] + seq[i]

                        # if the last character of the curr is end of formula character, then break
                        if curr[-1] == ';':
                            break
    
    # Convert to numpy arrays and then to torch tensors
    if len(features) >= min_valid_samples:
        features_np = np.stack(features)
        labels_np = np.array(labels)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        print(f"Feature shape: {features_tensor.shape}")
        print(f"Labels range: {labels_tensor.min().item()} to {labels_tensor.max().item()}")
        print(f"Using plain ASCII codes without remapping")
        
        if debug:
            print(f"SMILES processing summary:")
            print(f"  Total SMILES processed: {total_smiles}")
            print(f"  Valid SMILES: {valid_smiles}")
            print(f"  Invalid SMILES: {invalid_smiles}")
            print(f"    - RDKit parse errors: {rdkit_parse_errors}")
            print(f"    - Missing separators: {missing_separators}")
            print(f"    - Other errors: {other_errors}")
            
            if problematic_samples:
                print("\nSample problematic SMILES:")
                for error_type, sample in problematic_samples:
                    print(f"  [{error_type}] {sample}")
        
        return features_tensor, labels_tensor, label_chars
    else:
        if debug:
            print(f"SMILES processing summary:")
            print(f"  Total SMILES processed: {total_smiles}")
            print(f"  Valid SMILES: {valid_smiles}")
            print(f"  Invalid SMILES: {invalid_smiles}")
            print(f"    - RDKit parse errors: {rdkit_parse_errors}")
            print(f"    - Missing separators: {missing_separators}")
            print(f"    - Other errors: {other_errors}")
            
            if problematic_samples:
                print("\nSample problematic SMILES:")
                for error_type, sample in problematic_samples:
                    print(f"  [{error_type}] {sample}")
        raise ValueError(f"No samples were generated. Only {len(features)} valid samples found, minimum required: {min_valid_samples}. Check the input data.")
