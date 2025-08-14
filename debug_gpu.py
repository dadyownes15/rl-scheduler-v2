import torch
import sys
import platform
import subprocess
import os

def check_nvidia_smi():
    """Check if nvidia-smi is available and get GPU info"""
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], universal_newlines=True)
        print("\n=== NVIDIA-SMI Output ===")
        print(nvidia_smi)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n=== NVIDIA-SMI Error ===")
        print("nvidia-smi not found or returned an error.")
        print("This might mean NVIDIA drivers are not installed properly.")

def check_cuda_toolkit():
    """Check CUDA toolkit installation"""
    if hasattr(torch.version, 'cuda'):
        print("\n=== CUDA Toolkit ===")
        print(f"CUDA Version (PyTorch): {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"CUDA Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"GPU {i} Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("\nCUDA not available in PyTorch installation")

def check_pytorch_build():
    """Check PyTorch build information"""
    print("\n=== PyTorch Build Info ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch Debug Build: {torch.version.debug}")
    print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"Current CUDA Device Name: {torch.cuda.get_device_name(0)}")

def test_gpu_tensor():
    """Try to create and operate on a GPU tensor"""
    print("\n=== GPU Tensor Test ===")
    if torch.cuda.is_available():
        try:
            # Try to create a tensor on GPU
            x = torch.tensor([1.0, 2.0, 3.0], device='cuda')
            print("Successfully created tensor on GPU:", x)
            print("Tensor device:", x.device)
            
            # Try a simple operation
            y = x * 2
            print("Successfully performed operation on GPU")
            print("Result:", y)
        except Exception as e:
            print("Error when trying to use GPU:", str(e))
    else:
        print("Cannot test GPU tensor - CUDA not available")

def check_system_info():
    """Check system information"""
    print("\n=== System Information ===")
    print(f"Python Version: {sys.version}")
    print(f"Operating System: {platform.system()} {platform.version()}")
    print(f"CPU Architecture: {platform.machine()}")
    
    # Check environment variables related to CUDA
    cuda_path = os.environ.get('CUDA_PATH', 'Not set')
    cuda_home = os.environ.get('CUDA_HOME', 'Not set')
    print(f"\nCUDA Environment Variables:")
    print(f"CUDA_PATH: {cuda_path}")
    print(f"CUDA_HOME: {cuda_home}")

def main():
    print("="*50)
    print("GPU Debugging Information")
    print("="*50)
    
    check_system_info()
    check_pytorch_build()
    check_cuda_toolkit()
    check_nvidia_smi()
    test_gpu_tensor()
    
    print("\n" + "="*50)
    print("Debug Complete")
    print("="*50)

if __name__ == "__main__":
    main()
