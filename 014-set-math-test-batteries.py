#!/usr/bin/env python3
"""
Script to generate batteries of math prompts for testing.
Creates files like battery-addition.txt, battery-subtraction.txt, etc.
Supports both random generation and exhaustive digit-based generation.
"""

import os
import random
import argparse
import itertools
from datetime import datetime

def generate_addition_prompts(num_prompts=100, min_val=0, max_val=100):
    """Generate addition prompts in the format 'a+b='"""
    prompts = []
    for _ in range(num_prompts):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        prompts.append(f"{a}+{b}=")
    return prompts

def generate_subtraction_prompts(num_prompts=100, min_val=0, max_val=100, allow_negative=False):
    """Generate subtraction prompts in the format 'a-b='"""
    prompts = []
    for _ in range(num_prompts):
        if allow_negative:
            a = random.randint(min_val, max_val)
            b = random.randint(min_val, max_val)
        else:
            # Ensure a >= b to avoid negative results if not allowed
            b = random.randint(min_val, max_val)
            a = random.randint(b, max_val)
        prompts.append(f"{a}-{b}=")
    return prompts

def generate_multiplication_prompts(num_prompts=100, min_val=0, max_val=20):
    """Generate multiplication prompts in the format 'a*b='"""
    prompts = []
    for _ in range(num_prompts):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        prompts.append(f"{a}*{b}=")
    return prompts

def generate_division_prompts(num_prompts=100, min_val=1, max_val=100, integer_only=True):
    """Generate division prompts in the format 'a/b='"""
    prompts = []
    for _ in range(num_prompts):
        b = random.randint(min_val, max_val)
        if b == 0:  # Avoid division by zero
            b = 1
            
        if integer_only:
            # Generate a that is divisible by b for integer results
            multiplier = random.randint(min_val, max_val)
            a = b * multiplier
        else:
            a = random.randint(min_val, max_val)
            
        prompts.append(f"{a}/{b}=")
    return prompts

def generate_mixed_prompts(num_prompts=100, min_val=0, max_val=50):
    """Generate mixed operation prompts"""
    prompts = []
    operations = ['+', '-', '*', '/']
    
    for _ in range(num_prompts):
        op = random.choice(operations)
        
        if op == '+':
            a = random.randint(min_val, max_val)
            b = random.randint(min_val, max_val)
        elif op == '-':
            a = random.randint(min_val, max_val)
            b = random.randint(min_val, min(a, max_val))  # Ensure b <= a to avoid negative results
        elif op == '*':
            a = random.randint(min_val, min(20, max_val))  # Smaller range for multiplication
            b = random.randint(min_val, min(20, max_val))
        elif op == '/':
            b = random.randint(1, min(20, max_val))  # Avoid division by zero
            multiplier = random.randint(1, min(10, max_val))
            a = b * multiplier  # Ensure clean division
            
        prompts.append(f"{a}{op}{b}=")
    return prompts

def generate_exponentiation_prompts(num_prompts=50, base_min=0, base_max=10, exp_min=0, exp_max=4):
    """Generate exponentiation prompts in the format 'a^b='"""
    prompts = []
    for _ in range(num_prompts):
        a = random.randint(base_min, base_max)
        b = random.randint(exp_min, exp_max)
        prompts.append(f"{a}^{b}=")
    return prompts

def generate_square_root_prompts(num_prompts=50, min_val=0, max_val=100, perfect_squares_only=True):
    """Generate square root prompts in the format 'sqrt(a)='"""
    prompts = []
    for _ in range(num_prompts):
        if perfect_squares_only:
            # Generate perfect squares for cleaner results
            root = random.randint(min_val, int(max_val**0.5))
            a = root * root
        else:
            a = random.randint(min_val, max_val)
        prompts.append(f"sqrt({a})=")
    return prompts

def generate_complex_expressions(num_prompts=50, min_val=1, max_val=20):
    """Generate more complex expressions with multiple operations"""
    prompts = []
    for _ in range(num_prompts):
        # Generate expressions like (a+b)*c or a*(b-c) or (a+b)/(c+d)
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        c = random.randint(min_val, max_val)
        
        expression_types = [
            f"({a}+{b})*{c}=",
            f"{a}*({b}-{c})=",
            f"({a}+{b})/({c}+1)=",  # Adding 1 to avoid division by zero
            f"{a}+{b}*{c}=",
            f"{a}*{b}+{c}=",
            f"{a}*{b}-{c}=",
            f"({a}+{b})*({c}+1)="
        ]
        
        prompts.append(random.choice(expression_types))
    return prompts

def generate_numbers_with_digits(num_digits):
    """Generate all numbers with exactly num_digits digits"""
    if num_digits == 1:
        return list(range(0, 10))  # Include 0 for single-digit numbers
    
    start = 10 ** (num_digits - 1)
    end = (10 ** num_digits) - 1
    return list(range(start, end + 1))

def generate_numbers_up_to_digits(num_digits):
    """Generate all numbers with up to num_digits digits"""
    return list(range(0, 10 ** num_digits))

def generate_exhaustive_addition_prompts(digits_a, digits_b):
    """Generate all possible addition prompts for numbers with specified digits"""
    prompts = []
    
    # Handle special cases for digits
    if digits_a == 1:
        numbers_a = list(range(0, 10))  # Include 0-9 for single digit
    else:
        numbers_a = generate_numbers_with_digits(digits_a)
    
    if digits_b == 1:
        numbers_b = list(range(0, 10))  # Include 0-9 for single digit
    else:
        numbers_b = generate_numbers_with_digits(digits_b)
    
    # For very large sets, we might need to sample
    if len(numbers_a) * len(numbers_b) > 1000000:
        print(f"Warning: Exhaustive generation would create {len(numbers_a) * len(numbers_b)} prompts.")
        print("Sampling 1,000,000 random combinations instead.")
        prompts = []
        for _ in range(1000000):
            a = random.choice(numbers_a)
            b = random.choice(numbers_b)
            prompts.append(f"{a}+{b}=")
        return prompts
    
    # Generate all combinations
    for a in numbers_a:
        for b in numbers_b:
            prompts.append(f"{a}+{b}=")
    
    return prompts

def generate_exhaustive_subtraction_prompts(digits_a, digits_b, allow_negative=False):
    """Generate all possible subtraction prompts for numbers with specified digits"""
    prompts = []
    
    # Handle special cases for digits
    if digits_a == 1:
        numbers_a = list(range(0, 10))  # Include 0-9 for single digit
    else:
        numbers_a = generate_numbers_with_digits(digits_a)
    
    if digits_b == 1:
        numbers_b = list(range(0, 10))  # Include 0-9 for single digit
    else:
        numbers_b = generate_numbers_with_digits(digits_b)
    
    # For very large sets, we might need to sample
    if len(numbers_a) * len(numbers_b) > 1000000:
        print(f"Warning: Exhaustive generation would create {len(numbers_a) * len(numbers_b)} prompts.")
        print("Sampling 1,000,000 random combinations instead.")
        prompts = []
        for _ in range(1000000):
            a = random.choice(numbers_a)
            b = random.choice(numbers_b)
            if allow_negative or a >= b:
                prompts.append(f"{a}-{b}=")
        return prompts
    
    # Generate all combinations
    for a in numbers_a:
        for b in numbers_b:
            if allow_negative or a >= b:
                prompts.append(f"{a}-{b}=")
    
    return prompts

def generate_exhaustive_multiplication_prompts(digits_a, digits_b):
    """Generate all possible multiplication prompts for numbers with specified digits"""
    prompts = []
    
    # Handle special cases for digits
    if digits_a == 1:
        numbers_a = list(range(0, 10))  # Include 0-9 for single digit
    else:
        numbers_a = generate_numbers_with_digits(digits_a)
    
    if digits_b == 1:
        numbers_b = list(range(0, 10))  # Include 0-9 for single digit
    else:
        numbers_b = generate_numbers_with_digits(digits_b)
    
    # For very large sets, we might need to sample
    if len(numbers_a) * len(numbers_b) > 1000000:
        print(f"Warning: Exhaustive generation would create {len(numbers_a) * len(numbers_b)} prompts.")
        print("Sampling 1,000,000 random combinations instead.")
        prompts = []
        for _ in range(1000000):
            a = random.choice(numbers_a)
            b = random.choice(numbers_b)
            prompts.append(f"{a}*{b}=")
        return prompts
    
    # Generate all combinations
    for a in numbers_a:
        for b in numbers_b:
            prompts.append(f"{a}*{b}=")
    
    return prompts

def generate_exhaustive_division_prompts(digits_a, digits_b, integer_only=True):
    """Generate all possible division prompts for numbers with specified digits"""
    prompts = []
    
    # Handle special cases for digits
    if digits_a == 1:
        numbers_a = list(range(0, 10))  # Include 0-9 for single digit
    else:
        numbers_a = generate_numbers_with_digits(digits_a)
    
    if digits_b == 1:
        # Include 1-9 for single digit (avoid division by zero)
        numbers_b = list(range(1, 10))
    else:
        numbers_b = [n for n in generate_numbers_with_digits(digits_b) if n != 0]  # Avoid division by zero
    
    # For very large sets, we might need to sample
    if len(numbers_a) * len(numbers_b) > 1000000:
        print(f"Warning: Exhaustive generation would create {len(numbers_a) * len(numbers_b)} prompts.")
        print("Sampling 1,000,000 random combinations instead.")
        prompts = []
        for _ in range(1000000):
            a = random.choice(numbers_a)
            b = random.choice(numbers_b)
            if not integer_only or (a % b == 0):
                prompts.append(f"{a}/{b}=")
        return prompts
    
    # Generate all combinations
    for a in numbers_a:
        for b in numbers_b:
            if not integer_only or (a % b == 0):
                prompts.append(f"{a}/{b}=")
    
    return prompts

def generate_digit_pattern_numbers(pattern):
    """
    Generate numbers matching a specific digit pattern.
    Pattern format: 
    - 'd' represents any digit 0-9
    - A specific digit (0-9) represents that exact digit
    - 'n' represents any non-zero digit 1-9
    
    Examples:
    - 'dd': All 2-digit numbers (10-99)
    - 'n0': All 2-digit numbers ending in 0 (10, 20, 30...)
    - '1d': All 2-digit numbers starting with 1 (10-19)
    """
    if not pattern:
        return [0]
        
    # Generate all possible combinations based on the pattern
    digit_options = []
    for char in pattern:
        if char == 'd':
            digit_options.append(list(range(10)))  # Any digit 0-9
        elif char == 'n':
            digit_options.append(list(range(1, 10)))  # Any non-zero digit 1-9
        elif char.isdigit():
            digit_options.append([int(char)])  # Specific digit
        else:
            raise ValueError(f"Invalid pattern character: {char}. Use 'd' for any digit, 'n' for non-zero digit, or a specific digit 0-9.")
    
    # Generate all combinations
    numbers = []
    for digits in itertools.product(*digit_options):
        num = int(''.join(map(str, digits)))
        numbers.append(num)
    
    return numbers

def generate_pattern_based_prompts(operation, pattern_a, pattern_b, allow_negative=False, integer_division=True):
    """Generate prompts based on specific digit patterns for operands"""
    numbers_a = generate_digit_pattern_numbers(pattern_a)
    
    if operation == 'division':
        # Filter out zero for division to avoid division by zero
        numbers_b = [n for n in generate_digit_pattern_numbers(pattern_b) if n != 0]
        if not numbers_b:
            numbers_b = [1]  # Default to 1 if all numbers were filtered out
    else:
        numbers_b = generate_digit_pattern_numbers(pattern_b)
    
    prompts = []
    
    # For very large sets, we might need to sample
    if len(numbers_a) * len(numbers_b) > 1000000:
        print(f"Warning: Pattern-based generation would create {len(numbers_a) * len(numbers_b)} prompts.")
        print("Sampling 1,000,000 random combinations instead.")
        for _ in range(1000000):
            a = random.choice(numbers_a)
            b = random.choice(numbers_b)
            
            if operation == 'addition':
                prompts.append(f"{a}+{b}=")
            elif operation == 'subtraction':
                if allow_negative or a >= b:
                    prompts.append(f"{a}-{b}=")
            elif operation == 'multiplication':
                prompts.append(f"{a}*{b}=")
            elif operation == 'division':
                if not integer_division or (a % b == 0):
                    prompts.append(f"{a}/{b}=")
        
        return prompts
    
    # Generate all combinations
    for a in numbers_a:
        for b in numbers_b:
            if operation == 'addition':
                prompts.append(f"{a}+{b}=")
            elif operation == 'subtraction':
                if allow_negative or a >= b:
                    prompts.append(f"{a}-{b}=")
            elif operation == 'multiplication':
                prompts.append(f"{a}*{b}=")
            elif operation == 'division':
                if not integer_division or (a % b == 0):
                    prompts.append(f"{a}/{b}=")
    
    return prompts

def generate_all_combinations_by_digits(operation, max_digits_a, max_digits_b, allow_negative=False, integer_division=True):
    """Generate all combinations of operations for numbers up to max_digits"""
    prompts = []
    
    # Include 0 as a special case
    numbers_with_0_a = list(range(0, 10))
    numbers_with_0_b = list(range(0, 10))
    
    # First handle the special case of 0
    print(f"Generating {operation} prompts including 0...")
    for a in numbers_with_0_a:
        for b in numbers_with_0_b:
            if operation == 'addition':
                prompts.append(f"{a}+{b}=")
            elif operation == 'subtraction':
                if allow_negative or a >= b:
                    prompts.append(f"{a}-{b}=")
            elif operation == 'multiplication':
                prompts.append(f"{a}*{b}=")
            elif operation == 'division':
                if b != 0 and (not integer_division or (a % b == 0)):
                    prompts.append(f"{a}/{b}=")
    
    # Now handle all other digit combinations
    for digits_a in range(1, max_digits_a + 1):
        for digits_b in range(1, max_digits_b + 1):
            # Skip single digit numbers as we've already handled them (including 0)
            if digits_a == 1 and digits_b == 1:
                continue
                
            print(f"Generating {operation} prompts for {digits_a}-digit and {digits_b}-digit numbers...")
            
            if operation == 'addition':
                batch_prompts = generate_exhaustive_addition_prompts(digits_a, digits_b)
            elif operation == 'subtraction':
                batch_prompts = generate_exhaustive_subtraction_prompts(digits_a, digits_b, allow_negative)
            elif operation == 'multiplication':
                batch_prompts = generate_exhaustive_multiplication_prompts(digits_a, digits_b)
            elif operation == 'division':
                batch_prompts = generate_exhaustive_division_prompts(digits_a, digits_b, integer_division)
            
            prompts.extend(batch_prompts)
            
            # Safety check for very large sets
            if len(prompts) > 5000000:
                print(f"Warning: Generated over 5 million prompts. Truncating to prevent memory issues.")
                return prompts[:5000000]
    
    return prompts

def save_prompts_to_file(prompts, filename):
    """Save a list of prompts to a file"""
    with open(filename, 'w') as f:
        for prompt in prompts:
            f.write(f"{prompt}\n")
    print(f"Created {filename} with {len(prompts)} prompts")

def main():
    parser = argparse.ArgumentParser(description='Generate batteries of math prompts for testing')
    parser.add_argument('--count', type=int, default=100, help='Number of prompts per operation')
    parser.add_argument('--min', type=int, default=0, help='Minimum value for operands')
    parser.add_argument('--max', type=int, default=100, help='Maximum value for operands')
    parser.add_argument('--operations', type=str, default='all', 
                        help='Comma-separated list of operations to generate (addition,subtraction,multiplication,division,mixed,exponentiation,sqrt,complex) or "all"')
    parser.add_argument('--allow-negative', action='store_true', help='Allow negative results in subtraction')
    parser.add_argument('--non-integer-division', action='store_true', help='Allow non-integer division results')
    parser.add_argument('--timestamp', action='store_true', help='Add timestamp to filenames')
    parser.add_argument('--exhaustive', action='store_true', help='Use exhaustive digit-based generation instead of random')
    parser.add_argument('--digits-a', type=int, default=1, help='Number of digits for operand a in exhaustive generation')
    parser.add_argument('--digits-b', type=int, default=1, help='Number of digits for operand b in exhaustive generation')
    parser.add_argument('--pattern-a', type=str, help='Digit pattern for operand a (e.g., "dd" for all 2-digit numbers)')
    parser.add_argument('--pattern-b', type=str, help='Digit pattern for operand b (e.g., "n0" for 10,20,30...)')
    parser.add_argument('--all-combinations', action='store_true', help='Generate all combinations up to max digits')
    parser.add_argument('--max-digits-a', type=int, default=1, help='Maximum number of digits for operand a in all-combinations mode')
    parser.add_argument('--max-digits-b', type=int, default=1, help='Maximum number of digits for operand b in all-combinations mode')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the prompts before saving')
    parser.add_argument('--limit', type=int, help='Limit the number of prompts per file')
    
    args = parser.parse_args()
    
    # Determine which operations to generate
    all_operations = ['addition', 'subtraction', 'multiplication', 'division', 'mixed', 'exponentiation', 'sqrt', 'complex']
    if args.operations.lower() == 'all':
        operations = all_operations
    else:
        operations = [op.strip() for op in args.operations.split(',')]
        # Validate operations
        for op in operations[:]:  # Create a copy to iterate over while potentially modifying the original
            if op not in all_operations:
                print(f"Warning: Unknown operation '{op}'. Skipping.")
                operations.remove(op)
    
    timestamp = f"-{datetime.now().strftime('%Y%m%d_%H%M%S')}" if args.timestamp else ""
    
    # Generate and save prompts for each selected operation
    for operation in operations:
        prompts = []
        
        # Pattern-based generation takes precedence if patterns are provided
        if args.pattern_a or args.pattern_b:
            pattern_a = args.pattern_a or 'd'  # Default to single digit if not specified
            pattern_b = args.pattern_b or 'd'  # Default to single digit if not specified
            
            print(f"Generating {operation} prompts with patterns: a='{pattern_a}', b='{pattern_b}'")
            prompts = generate_pattern_based_prompts(
                operation, 
                pattern_a, 
                pattern_b, 
                args.allow_negative, 
                not args.non_integer_division
            )
            
            if prompts:
                if args.shuffle:
                    random.shuffle(prompts)
                if args.limit and len(prompts) > args.limit:
                    prompts = prompts[:args.limit]
                save_prompts_to_file(prompts, f"battery-{operation}-pattern{timestamp}.txt")
            continue
            
        # All combinations mode
        if args.all_combinations and operation in ['addition', 'subtraction', 'multiplication', 'division']:
            print(f"Generating all combinations of {operation} for numbers up to {args.max_digits_a} and {args.max_digits_b} digits")
            prompts = generate_all_combinations_by_digits(
                operation,
                args.max_digits_a,
                args.max_digits_b,
                args.allow_negative,
                not args.non_integer_division
            )
            
            if prompts:
                if args.shuffle:
                    random.shuffle(prompts)
                if args.limit and len(prompts) > args.limit:
                    prompts = prompts[:args.limit]
                save_prompts_to_file(prompts, f"battery-{operation}-all-combinations{timestamp}.txt")
            continue
        
        # Exhaustive digit-based generation
        if args.exhaustive:
            if operation in ['addition', 'subtraction', 'multiplication', 'division']:
                if operation == 'addition':
                    prompts = generate_exhaustive_addition_prompts(args.digits_a, args.digits_b)
                elif operation == 'subtraction':
                    prompts = generate_exhaustive_subtraction_prompts(args.digits_a, args.digits_b, args.allow_negative)
                elif operation == 'multiplication':
                    prompts = generate_exhaustive_multiplication_prompts(args.digits_a, args.digits_b)
                elif operation == 'division':
                    prompts = generate_exhaustive_division_prompts(args.digits_a, args.digits_b, not args.non_integer_division)
                
                if prompts:
                    if args.shuffle:
                        random.shuffle(prompts)
                    if args.limit and len(prompts) > args.limit:
                        prompts = prompts[:args.limit]
                    save_prompts_to_file(prompts, f"battery-{operation}-exhaustive{timestamp}.txt")
            else:
                print(f"Warning: Exhaustive generation not supported for operation '{operation}'. Skipping.")
            continue
        
        # Default random generation
        if operation == 'addition':
            prompts = generate_addition_prompts(args.count, args.min, args.max)
        elif operation == 'subtraction':
            prompts = generate_subtraction_prompts(args.count, args.min, args.max, args.allow_negative)
        elif operation == 'multiplication':
            # Use a smaller max value for multiplication to avoid huge numbers
            max_val = min(args.max, 20)
            prompts = generate_multiplication_prompts(args.count, args.min, max_val)
        elif operation == 'division':
            # Use a smaller max value for division
            max_val = min(args.max, 100)
            prompts = generate_division_prompts(args.count, max(args.min, 1), max_val, not args.non_integer_division)
        elif operation == 'mixed':
            prompts = generate_mixed_prompts(args.count, args.min, args.max)
        elif operation == 'exponentiation':
            # Use smaller ranges for exponentiation
            base_max = min(args.max, 10)
            exp_max = min(args.max, 4)
            prompts = generate_exponentiation_prompts(args.count, args.min, base_max, 0, exp_max)
        elif operation == 'sqrt':
            prompts = generate_square_root_prompts(args.count, max(args.min, 0), args.max)
        elif operation == 'complex':
            # Use smaller ranges for complex expressions
            max_val = min(args.max, 20)
            prompts = generate_complex_expressions(args.count, max(args.min, 1), max_val)
        
        if prompts:
            if args.shuffle:
                random.shuffle(prompts)
            if args.limit and len(prompts) > args.limit:
                prompts = prompts[:args.limit]
            save_prompts_to_file(prompts, f"battery-{operation}{timestamp}.txt")

if __name__ == "__main__":
    main()
