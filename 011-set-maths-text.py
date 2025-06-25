#!/usr/bin/env python3
"""
Script to create a text dataset of arithmetic problems with random numbers.
The script generates problems in the formats: A+B=C; A-B=C; A*B=C;
with increasing ranges of random numbers.
Each combination of operation and numbers occurs only once across the entire dataset.
The "=" character is padded to be exactly at the LEARNING_CONTEXT position in each line by adding spaces before the expression.
Each line is padded to a total length of 32 characters.
The dataset is split into training (85%), validation (10%), and testing (5%) files.
"""

import os
import argparse

LEARNING_CONTEXT = 16
TOTAL_LINE_LENGTH = 32

CUT_LINE_AT_SEMICOLON = True

def generate_arithmetic_dataset(operations_to_include=None, max_digits=3, sample_limits=None):
    """
    Generate a dataset of arithmetic problems with all possible number combinations in order.
    
    Args:
        operations_to_include (list, optional): List of operations to include ('+', '-', '*').
                                               Default is all operations.
        max_digits (int, optional): Maximum number of digits for the numbers.
                                   Will generate datasets for 1 up to max_digits.
                                   Default is 3 (up to 3-digit numbers).
        sample_limits (dict, optional): Dictionary with digit counts as keys and sample limits as values.
                                       If not provided, defaults to n^2 where n is 10^digits.
    
    Returns:
        list: All generated lines
    """
    # Set to track unique combinations across all ranges and operations
    unique_combinations = set()
    all_lines = []
    
    # Operations to generate
    all_operations = [
        ('+', lambda a, b: a + b),  # Addition
        ('-', lambda a, b: a - b),  # Subtraction
        ('*', lambda a, b: a * b)   # Multiplication
    ]
    
    # Filter operations based on input
    if operations_to_include:
        operations = [(symbol, func) for symbol, func in all_operations if symbol in operations_to_include]
    else:
        operations = all_operations
    
    if not operations:
        print("Warning: No valid operations specified. Using all operations.")
        operations = all_operations
    
    print(f"Generating dataset with operations: {[op[0] for op in operations]}")
    
    # Initialize sample_limits if not provided
    if sample_limits is None:
        sample_limits = {}
    
    # Generate datasets for each digit count from 1 to max_digits
    for digits in range(1, max_digits + 1):
        max_num = 10 ** digits - 1  # e.g., 9 for 1 digit, 99 for 2 digits, 999 for 3 digits
        
        # Default sample limit is the square of the range size, but can be overridden
        default_limit = min((10 ** digits) ** 2, 1000000)  # Cap at 1 million to avoid excessive generation
        target_count = sample_limits.get(digits, default_limit)
        
        print(f"Generating up to {target_count} unique examples with {digits}-digit numbers (0 to {max_num})...")
        
        start_count = len(unique_combinations)
        digit_target = target_count  # Store the target specifically for this digit range
        generated_count = 0
        
        # Iterate through all operations first
        for op_symbol, op_func in operations:
            # Then iterate through all possible values of a
            for a in range(max_num + 1):
                # Then iterate through all possible values of b
                b_range = min(99, max_num) if op_symbol == '*' and digits > 2 else max_num
                
                for b in range(b_range + 1):
                    # For subtraction, ensure a >= b to avoid negative results
                    if op_symbol == '-' and a < b:
                        continue
                    
                    # Calculate the result
                    c = op_func(a, b)
                    
                    # Create a unique key for this combination
                    combo_key = f"{a}{op_symbol}{b}"
                    
                    # Only add if this is a new combination and we haven't reached the target
                    if combo_key not in unique_combinations and generated_count < digit_target:
                        unique_combinations.add(combo_key)
                        generated_count += 1
                        
                        # Format the line with proper padding
                        expression = f"{a}{op_symbol}{b}"
                        # Calculate padding needed to position the "=" at the LEARNING_CONTEXT character
                        padding_length = LEARNING_CONTEXT - len(expression) - 1
                        # Create the line with the equals sign at the right position
                        line = " " * padding_length + expression + "=" + str(c) + ";"
                        if CUT_LINE_AT_SEMICOLON:
                            line = line[:TOTAL_LINE_LENGTH]
                        else:
                            line = line.ljust(TOTAL_LINE_LENGTH)
                        all_lines.append(line)
                    
                    # If we've reached the target, break out of the innermost loop
                    if generated_count >= digit_target:
                        break
                
                # If we've reached the target, break out of the middle loop
                if generated_count >= digit_target:
                    break
            
            # If we've reached the target, break out of the outermost loop
            if generated_count >= digit_target:
                break
        
        actual_generated = generated_count
        print(f"Generated {actual_generated} unique combinations with {digits}-digit numbers")
    
    total_generated = len(unique_combinations)
    print(f"Total dataset size: {total_generated} unique combinations")
    
    return all_lines

def split_and_save_dataset(all_lines, output_file):
    """
    Save the dataset to a file.
    
    Args:
        all_lines (list): All generated lines
        output_file (str): Path to the output file
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Delete old file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Write the file
    with open(output_file, 'w') as f:
        f.write('\n'.join(all_lines))
    
    print(f"Dataset saved to {output_file}")

def main():
    # Create a parser with a more descriptive help message
    parser = argparse.ArgumentParser(
        description='Arithmetic dataset generator with custom parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Show this help message
  python %(prog)s
  
  # Generate a dataset with all operations (addition, subtraction, multiplication)
  python %(prog)s --generate
  
  # Generate a dataset with only addition and subtraction
  python %(prog)s --generate --operations "+,-"
  
  # Generate a dataset with 2-digit numbers and custom output directory
  python %(prog)s --generate --max-digits 2 --output-dir "my_data/arithmetic"
  
  # Generate a smaller dataset with custom limits
  python %(prog)s --generate --digit-1-limit 50 --digit-2-limit 500
'''
    )
    
    # Required action group
    action_group = parser.add_argument_group('Action (Required)')
    action_group.add_argument('--generate', action='store_true',
                        help='Generate the dataset (required to perform any generation)')
    
    # Group for operation configuration
    op_group = parser.add_argument_group('Operation Configuration')
    op_group.add_argument('--operations', type=str, default='+,-,*', 
                        help='Comma-separated list of operations to include (default: "+,-,*")')
    
    # Group for dataset size configuration
    size_group = parser.add_argument_group('Dataset Size Configuration')
    size_group.add_argument('--max-digits', type=int, default=3,
                        help='Maximum number of digits for the numbers (default: 3)')
    size_group.add_argument('--digit-1-limit', type=int, default=10*10,
                        help='Maximum number of examples for 1-digit numbers (default: 100)')
    size_group.add_argument('--digit-2-limit', type=int, default=100*100,
                        help='Maximum number of examples for 2-digit numbers (default: 10000)')
    size_group.add_argument('--digit-3-limit', type=int, default=1000*1000,
                        help='Maximum number of examples for 3-digit numbers (default: 1000000)')
    size_group.add_argument('--digit-4-limit', type=int, default=10000*10000,
                        help='Maximum number of examples for 4-digit numbers (default: 100000000)')
    
    # Group for output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output', type=str, default="./dataset_011-maths-text.txt",
                        help='Output file path to save the dataset (default: "./dataset_011-maths-text.txt")')
    
    args = parser.parse_args()
    
    # If generate flag is not provided, just show help and exit
    if not args.generate:
        parser.print_help()
        print("\nNOTE: Use the --generate flag to generate the dataset.")
        return
    
    # Parse operations from command line
    operations = [op.strip() for op in args.operations.split(',') if op.strip() in ['+', '-', '*']]
    
    # Create sample limits dictionary
    sample_limits = {
        1: args.digit_1_limit,
        2: args.digit_2_limit,
        3: args.digit_3_limit,
        4: args.digit_4_limit
    }
    
    print("Starting dataset generation with the following parameters:")
    print(f"  Operations: {operations}")
    print(f"  Max digits: {args.max_digits}")
    print(f"  Output file: {args.output}")
    print(f"  Sample limits: 1-digit: {args.digit_1_limit}, 2-digit: {args.digit_2_limit}, 3-digit: {args.digit_3_limit}")
    print("\nGenerating dataset...")
    
    # Generate the dataset with specified parameters
    all_lines = generate_arithmetic_dataset(
        operations_to_include=operations,
        max_digits=args.max_digits,
        sample_limits=sample_limits
    )
    
    # Split and save the dataset
    output_file = args.output
    split_and_save_dataset(all_lines, output_file)
    
    print("Dataset generation completed successfully!")
    print(f"Output saved to: {output_file}")
    print(f"Total examples generated: {len(all_lines)}")


if __name__ == "__main__":
    main()
