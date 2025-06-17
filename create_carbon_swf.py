#!/usr/bin/env python3
"""
Script to create a new SWF file from lublin_256.swf with carbon consideration indices.
The carbon consideration can be either discrete (0-4) or float (0.0-1.0) based on user choice.
"""

import random
import os
import argparse

def create_carbon_swf(use_float=False, min_val=0.0, max_val=1.0):
    """
    Read lublin_256.swf and create lublin_256_carbon.swf with carbon consideration indices
    
    Args:
        use_float: If True, generate float values. If False, generate discrete integers (0-4)
        min_val: Minimum value for float generation (default: 0.0)
        max_val: Maximum value for float generation (default: 1.0)
    """
    input_file = "./data/lublin_256.swf"
    
    # Different output files for discrete vs float
    if use_float:
        output_file = "./data/lublin_256_carbon_float.swf"
    else:
        output_file = "./data/lublin_256_carbon.swf"
    
    # Set random seed for reproducible carbon consideration assignment
    random.seed(42)
    
    if use_float:
        print(f"Reading from: {input_file}")
        print(f"Writing to: {output_file}")
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
        
    else:
        # Original discrete implementation
        # Distribution weights for carbon consideration levels
        # This creates a realistic distribution where most users have moderate concern
        carbon_weights = {
            0: 0.15,  # 15% - Very low carbon concern
            1: 0.25,  # 25% - Low carbon concern  
            2: 0.30,  # 30% - Medium carbon concern
            3: 0.20,  # 20% - High carbon concern
            4: 0.10   # 10% - Very high carbon concern
        }
        
        # Create weighted list for random selection
        carbon_choices = []
        for level, weight in carbon_weights.items():
            carbon_choices.extend([level] * int(weight * 100))
        
        print(f"Reading from: {input_file}")
        print(f"Writing to: {output_file}")
        print("Carbon consideration: DISCRETE values (0-4)")
        print("Distribution:")
        for level, weight in carbon_weights.items():
            print(f"  Level {level}: {weight*100:5.1f}% (concern: {'very low' if level==0 else 'low' if level==1 else 'medium' if level==2 else 'high' if level==3 else 'very high'})")
        
        job_count = 0
        carbon_stats = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                line = line.strip()
                
                # Copy header/comment lines as-is
                if line.startswith(';') or line == '':
                    outfile.write(line + '\n')
                    continue
                
                # For job lines, append carbon consideration index
                carbon_level = random.choice(carbon_choices)
                carbon_stats[carbon_level] += 1
                job_count += 1
                
                # Append carbon consideration as the last field
                new_line = line + ' ' + str(carbon_level)
                outfile.write(new_line + '\n')
        
        print(f"\nProcessed {job_count} jobs")
        print("Actual carbon consideration distribution:")
        for level in range(5):
            percentage = (carbon_stats[level] / job_count) * 100
            print(f"  Level {level}: {carbon_stats[level]:5d} jobs ({percentage:5.1f}%)")
    
    print(f"\nSuccessfully created: {output_file}")
    return output_file

def verify_carbon_swf(filename, use_float=False):
    """
    Verify the created SWF file has correct format and carbon indices/values
    
    Args:
        filename: SWF file to verify
        use_float: If True, expect float values. If False, expect discrete integers (0-4)
    """
    print(f"\nVerifying {filename}...")
    
    job_count = 0
    
    if use_float:
        carbon_values = []
    else:
        carbon_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    
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
                if use_float:
                    carbon_value = float(fields[-1])
                    carbon_values.append(carbon_value)
                else:
                    carbon_index = int(fields[-1])
                    if carbon_index not in [0, 1, 2, 3, 4]:
                        print(f"ERROR: Line {line_num} has invalid carbon index: {carbon_index}")
                        return False
                    carbon_counts[carbon_index] += 1
                job_count += 1
            except ValueError:
                value_type = "float" if use_float else "integer"
                print(f"ERROR: Line {line_num} has non-{value_type} carbon value: {fields[-1]}")
                return False
    
    print(f"✓ Verified {job_count} jobs")
    
    if use_float:
        print("Carbon value statistics:")
        print(f"  Min value: {min(carbon_values):.6f}")
        print(f"  Max value: {max(carbon_values):.6f}")
        print(f"  Mean value: {sum(carbon_values)/len(carbon_values):.6f}")
    else:
        print("Carbon index distribution:")
        for level in range(5):
            percentage = (carbon_counts[level] / job_count) * 100
            print(f"  Level {level}: {carbon_counts[level]:5d} jobs ({percentage:5.1f}%)")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Carbon-Aware SWF File")
    parser.add_argument('--float', action='store_true', 
                       help='Generate float carbon consideration values instead of discrete (0-4)')
    parser.add_argument('--min', type=float, default=0.0,
                       help='Minimum value for float generation (default: 0.0)')
    parser.add_argument('--max', type=float, default=1.0,
                       help='Maximum value for float generation (default: 1.0)')
    
    args = parser.parse_args()
    
    print("Creating Carbon-Aware SWF File")
    print("=" * 50)
    
    # Validate arguments
    if args.float and args.min >= args.max:
        print("ERROR: --min must be less than --max")
        exit(1)
    
    # Check if input file exists
    if not os.path.exists("./data/lublin_256.swf"):
        print("ERROR: ./data/lublin_256.swf not found!")
        exit(1)
    
    # Create the new SWF file
    output_file = create_carbon_swf(use_float=args.float, min_val=args.min, max_val=args.max)
    
    # Verify the created file
    if verify_carbon_swf(output_file, use_float=args.float):
        print("\n✅ Carbon SWF file created and verified successfully!")
        
        print("\nTo use the new file, update your code to:")
        print(f"  env.my_init('{output_file}')")
        print("  or")
        print(f"  HPCEnv().my_init('{output_file}')")
        
        if args.float:
            print("\n⚠️  Note: When using float carbon values, you may need to update")
            print("   the job.py file to handle float parsing and normalization.")
    else:
        print("\n❌ Verification failed!")
        exit(1) 