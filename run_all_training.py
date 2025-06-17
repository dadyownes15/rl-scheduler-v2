#!/usr/bin/env python3
"""
Training script for all four RL algorithms on carbon-aware workload
Runs: MaskablePPO, MaskablePPO_Carbon, MARL, MARL_Plus

Cross-platform compatible (Windows, macOS, Linux)
"""

import subprocess
import sys
import time
import os
import platform

def get_python_executable():
    """Get the correct Python executable for the current platform"""
    if platform.system() == "Windows":
        # On Windows, prefer 'python' over 'python3'
        return "python"
    else:
        # On Unix-like systems, prefer 'python3' if available
        try:
            subprocess.run([sys.executable, "--version"], check=True, capture_output=True)
            return sys.executable
        except:
            return "python3"

def run_algorithm(algorithm_name, script_name, workload, epochs, backfill):
    """Run a single algorithm training"""
    print(f"\n{'='*60}")
    print(f"Starting {algorithm_name} training...")
    print(f"Workload: {workload}")
    print(f"Epochs: {epochs}")
    print(f"Backfill: {backfill}")
    print(f"Platform: {platform.system()}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the training script
        cmd = [
            get_python_executable(), script_name,
            "--workload", workload,
            "--backfill", str(backfill)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Platform-specific subprocess settings
        if platform.system() == "Windows":
            # On Windows, we need to handle the console properly
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                encoding='utf-8',
                errors='replace'  # Handle any encoding issues gracefully
            )
        else:
            result = subprocess.run(
                cmd, 
                check=True, 
                capture_output=True, 
                text=True,
                encoding='utf-8',
                errors='replace'
            )
        
        # Print any output
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n✅ {algorithm_name} completed successfully!")
        print(f"Training time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n❌ {algorithm_name} failed!")
        print(f"Error code: {e.returncode}")
        print(f"Training time before failure: {duration:.2f} seconds")
        
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
            
        return False
    
    except KeyboardInterrupt:
        print(f"\n⚠️ {algorithm_name} interrupted by user!")
        return False

def modify_epochs_in_file(filename, new_epochs):
    """Temporarily modify the epochs value in a training file"""
    print(f"Modifying epochs to {new_epochs} in {filename}...")
    
    # Read the file with UTF-8 encoding
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace epochs = 300 with epochs = new_epochs
    modified_content = content.replace('epochs = 300', f'epochs = {new_epochs}')
    
    # Write back to file with UTF-8 encoding
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✅ Modified {filename} to use {new_epochs} epochs")

def restore_epochs_in_file(filename, original_epochs=300):
    """Restore the original epochs value in a training file"""
    print(f"Restoring epochs to {original_epochs} in {filename}...")
    
    # Read the file with UTF-8 encoding
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find current epochs value and replace with original
    import re
    modified_content = re.sub(r'epochs = \d+', f'epochs = {original_epochs}', content)
    
    # Write back to file with UTF-8 encoding
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(modified_content)
    
    print(f"✅ Restored {filename} to use {original_epochs} epochs")

def check_system_requirements():
    """Check if the system meets the requirements for training"""
    print("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print(f"❌ ERROR: Python 3.7+ required. Current version: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check platform
    system = platform.system()
    print(f"✅ Platform: {system} {platform.release()}")
    
    # Check if required modules can be imported
    required_modules = ['torch', 'numpy', 'pandas']
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} available")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} not found")
    
    if missing_modules:
        print(f"\n❌ ERROR: Missing required modules: {', '.join(missing_modules)}")
        print("Please install them using:")
        if platform.system() == "Windows":
            print(f"pip install {' '.join(missing_modules)}")
        else:
            print(f"pip3 install {' '.join(missing_modules)}")
        return False
    
    print("✅ All system requirements met!")
    return True

def main():
    # Configuration
    workload = "lublin_256_carbon_float"
    epochs = 60
    backfill = 2
    
    # Algorithm configurations: (name, script_file)
    algorithms = [
        ("MaskablePPO", "MaskablePPO.py"),
        ("MaskablePPO_Carbon", "MaskablePPO_Carbon.py"),
        ("MARL", "MARL.py"),
        ("MARL_Plus", "MARL_Plus.py")
    ]
    
    print("="*80)
    print("CARBON-AWARE RL SCHEDULER TRAINING")
    print("="*80)
    
    # Check system requirements first
    if not check_system_requirements():
        print("\n❌ System requirements not met. Exiting...")
        sys.exit(1)
    
    print(f"\nWorkload: {workload}")
    print(f"Epochs per algorithm: {epochs}")
    print(f"Backfill mode: {backfill}")
    print(f"Algorithms to train: {len(algorithms)}")
    for i, (name, _) in enumerate(algorithms, 1):
        print(f"  {i}. {name}")
    print("="*80)
    
    # Check if workload file exists (cross-platform path handling)
    workload_file = os.path.join("data", f"{workload}.swf")
    if not os.path.exists(workload_file):
        print(f"❌ ERROR: Workload file not found: {workload_file}")
        print("Please make sure you have generated the carbon-aware workload first.")
        if platform.system() == "Windows":
            print("Run: python create_carbon_swf.py --float --workload lublin_256")
        else:
            print("Run: python3 create_carbon_swf.py --float --workload lublin_256")
        sys.exit(1)
    
    # Check if all algorithm files exist
    missing_files = []
    for name, script in algorithms:
        if not os.path.exists(script):
            missing_files.append(script)
    
    if missing_files:
        print(f"❌ ERROR: Missing algorithm files:")
        for file in missing_files:
            print(f"  - {file}")
        sys.exit(1)
    
    # Modify epochs in all files
    script_files = [script for _, script in algorithms]
    for script in script_files:
        modify_epochs_in_file(script, epochs)
    
    # Track results
    results = {}
    total_start_time = time.time()
    
    try:
        # Run each algorithm
        for i, (name, script) in enumerate(algorithms, 1):
            print(f"\n\n🚀 Starting algorithm {i}/{len(algorithms)}: {name}")
            
            success = run_algorithm(name, script, workload, epochs, backfill)
            results[name] = success
            
            # Brief pause between algorithms
            if i < len(algorithms):
                print(f"\nWaiting 5 seconds before starting next algorithm...")
                time.sleep(5)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user!")
    
    finally:
        # Restore original epochs in all files
        print(f"\n\n🔄 Restoring original epochs (300) in all files...")
        for script in script_files:
            try:
                restore_epochs_in_file(script, 300)
            except Exception as e:
                print(f"⚠️ Warning: Could not restore {script}: {e}")
    
    # Print final summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    print("\n\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total training time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    print(f"Workload: {workload}")
    print(f"Epochs per algorithm: {epochs}")
    print(f"Backfill mode: {backfill}")
    print("\nResults:")
    
    successful = 0
    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {name}: {status}")
        if success:
            successful += 1
    
    print(f"\nSummary: {successful}/{len(algorithms)} algorithms completed successfully")
    
    if successful == len(algorithms):
        print("\n🎉 All algorithms completed successfully!")
        print("\nNext steps:")
        print("1. Check the CSV files for training metrics:")
        for name, _ in algorithms:
            if name == "MaskablePPO":
                csv_name = f"MaskablePPO_{workload}.csv"
            elif name == "MaskablePPO_Carbon":
                csv_name = f"MaskablePPO_Carbon_{workload}.csv"
            elif name == "MARL":
                csv_name = f"MARL_{workload}.csv"
            elif name == "MARL_Plus":
                csv_name = f"MARL_Plus_{workload}.csv"
            print(f"   - {csv_name}")
        
        print("\n2. Check the saved models in:")
        print(f"   - {os.path.join(workload, 'MaskablePPO')}")
        print(f"   - {os.path.join(workload, 'MaskablePPO_Carbon')}")
        print(f"   - {os.path.join(workload, 'MARL')}")
        print(f"   - {os.path.join(workload, 'MARL_Plus')}")
        
        print(f"\n3. Run comparison:")
        if platform.system() == "Windows":
            print(f"   python compare.py --workload {workload} --len 1024 --iter 10 --backfill {backfill}")
        else:
            print(f"   python3 compare.py --workload {workload} --len 1024 --iter 10 --backfill {backfill}")
    
    elif successful > 0:
        print(f"\n⚠️ Partial success: {successful} out of {len(algorithms)} algorithms completed.")
        print("You may want to re-run the failed algorithms individually.")
    
    else:
        print("\n💥 All algorithms failed. Please check the error messages above.")
    
    print("="*80)

if __name__ == "__main__":
    main() 