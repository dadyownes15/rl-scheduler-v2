#!/usr/bin/env python3
"""
Script to create a new SWF file from lublin_256.swf with carbon consideration indices.
The carbon consideration can be either discrete (0-4) or float (0.0-1.0) based on user choice.
"""

import random
import os
import argparse

def create_carbon_swf(min_val=0.0, max_val=1.0, generation_mode='uniform'):
    """
    Read lublin_256.swf and create lublin_256_carbon.swf with float carbon consideration values
    
    Args:
        min_val: Minimum value for float generation (default: 0.0)
        max_val: Maximum value for float generation (default: 1.0)
        generation_mode: 'uniform' for uniform distribution, 'simple' for simple 3-level distribution
    """
    input_file = "./data/lublin_256.swf"
    
    # Different output files based on generation mode
    if generation_mode == 'simple':
        output_file = "./data/lublin_256_carbon_float_simple.swf"
    else:
        output_file = "./data/lublin_256_carbon_float.swf"
    
    # Set random seed for reproducible carbon consideration assignment
    random.seed(42)
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    if generation_mode == 'simple':
        print("Carbon consideration: SIMPLE mode with 3 discrete levels")
        print("Distribution:")
        print("  0.0: 80% (no carbon concern)")
        print("  0.5: 10% (medium carbon concern)")
        print("  1.0: 10% (high carbon concern)")
        
        # Create weighted choices for simple mode
        carbon_choices = [0.0] * 80 + [0.5] * 10 + [1.0] * 10
        
        job_count = 0
        carbon_stats = {0.0: 0, 0.5: 0, 1.0: 0}
        
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                line = line.strip()
                
                # Copy header/comment lines as-is
                if line.startswith(';') or line == '':
                    outfile.write(line + '\n')
                    continue
                
                # For job lines, append carbon consideration value from simple distribution
                carbon_value = random.choice(carbon_choices)
                carbon_stats[carbon_value] += 1
                job_count += 1
                
                # Append carbon consideration as the last field (formatted to 1 decimal place for simple mode)
                new_line = line + ' ' + f"{carbon_value:.1f}"
                outfile.write(new_line + '\n')
        
        print(f"\nProcessed {job_count} jobs")
        print("Actual carbon consideration distribution:")
        for level in [0.0, 0.5, 1.0]:
            percentage = (carbon_stats[level] / job_count) * 100
            print(f"  {level:.1f}: {carbon_stats[level]:5d} jobs ({percentage:5.1f}%)")
    else:
        print(f"Carbon consideration: FLOAT values uniformly distributed between {min_val} and {max_val}")
        
        job_count = 0
        carbon_values = []
        
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                line = line.strip()
                
                # Copy header/comment lines as-is
                if line.startswith(';') or line == '':
                    outfile.write(line + '\n')
                    continue
                
                # For job lines, append carbon consideration float value
                carbon_value = random.uniform(min_val, max_val)
                carbon_values.append(carbon_value)
                job_count += 1
                
                # Append carbon consideration as the last field (formatted to 6 decimal places)
                new_line = line + ' ' + f"{carbon_value:.6f}"
                outfile.write(new_line + '\n')
        
        print(f"\nProcessed {job_count} jobs")
        print(f"Carbon consideration statistics:")
        print(f"  Min value: {min(carbon_values):.6f}")
        print(f"  Max value: {max(carbon_values):.6f}")
        print(f"  Mean value: {sum(carbon_values)/len(carbon_values):.6f}")
    
    print(f"\nSuccessfully created: {output_file}")
    return output_file

def verify_carbon_swf(filename, generation_mode='uniform'):
    """
    Verify the created SWF file has correct format and carbon float values
    
    Args:
        filename: SWF file to verify
        generation_mode: 'uniform' for uniform distribution, 'simple' for simple 3-level distribution
    """
    print(f"\nVerifying {filename}...")
    
    job_count = 0
    
    if generation_mode == 'simple':
        carbon_stats = {0.0: 0, 0.5: 0, 1.0: 0}
    else:
        carbon_values = []
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip header/comment lines
            if line.startswith(';') or line == '':
                continue
                
            # Parse job line
            fields = line.split()
            if len(fields) < 19:  # Original 18 fields + 1 carbon field
                print(f"ERROR: Line {line_num} has only {len(fields)} fields, expected 19")
                return False
            
            # Check carbon consideration value (last field)
            try:
                carbon_value = float(fields[-1])
                if generation_mode == 'simple':
                    if carbon_value not in [0.0, 0.5, 1.0]:
                        print(f"ERROR: Line {line_num} has invalid carbon value for simple mode: {carbon_value}")
                        return False
                    carbon_stats[carbon_value] += 1
                else:
                    carbon_values.append(carbon_value)
                job_count += 1
            except ValueError:
                print(f"ERROR: Line {line_num} has non-float carbon value: {fields[-1]}")
                return False
    
    print(f"✓ Verified {job_count} jobs")
    
    if generation_mode == 'simple':
        print("Carbon value distribution (simple mode):")
        for level in [0.0, 0.5, 1.0]:
            percentage = (carbon_stats[level] / job_count) * 100
            print(f"  {level:.1f}: {carbon_stats[level]:5d} jobs ({percentage:5.1f}%)")
    else:
        print("Carbon value statistics:")
        print(f"  Min value: {min(carbon_values):.6f}")
        print(f"  Max value: {max(carbon_values):.6f}")
        print(f"  Mean value: {sum(carbon_values)/len(carbon_values):.6f}")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Carbon-Aware SWF File")
    parser.add_argument('--min', type=float, default=0.0,
                       help='Minimum value for float generation (default: 0.0)')
    parser.add_argument('--max', type=float, default=1.0,
                       help='Maximum value for float generation (default: 1.0)')
    parser.add_argument('--generation_mode', choices=['uniform', 'simple'], default='uniform',
                       help='Generation mode: uniform (default) or simple (80%% zero, 10%% at 0.5, 10%% at 1.0)')
    
    args = parser.parse_args()
    
    print("Creating Carbon-Aware SWF File")
    print("=" * 50)
    
    # Validate arguments
    if args.min >= args.max:
        print("ERROR: --min must be less than --max")
        exit(1)
    
    # For simple mode, ignore min/max values
    if args.generation_mode == 'simple':
        if args.min != 0.0 or args.max != 1.0:
            print("Note: Simple mode uses fixed values (0.0, 0.5, 1.0), ignoring --min and --max")
    
    # Check if input file exists
    if not os.path.exists("./data/lublin_256.swf"):
        print("ERROR: ./data/lublin_256.swf not found!")
        exit(1)
    
    # Create the new SWF file
    output_file = create_carbon_swf(min_val=args.min, max_val=args.max, generation_mode=args.generation_mode)
    
    # Verify the created file
    if verify_carbon_swf(output_file, generation_mode=args.generation_mode):
        print("\n✅ Carbon SWF file created and verified successfully!")
        
        print("\nTo use the new file, update your code to:")
        print(f"  env.my_init('{output_file}')")
        print("  or")
        print(f"  HPCEnv().my_init('{output_file}')")
        
        print("\n⚠️  Note: All carbon values are float values between 0.0 and 1.0")
    else:
        print("\n❌ Verification failed!")
        exit(1) 