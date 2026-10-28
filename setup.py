import os
import sys
import torch

def check_environment():
    """Verifies Python, PyTorch, and CUDA/GPU status."""
    print("=" * 50)
    print("       CV PROJECT ENVIRONMENT DIAGNOSTIC")
    print("=" * 50)
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA (GPU Acceleration) Available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")
        # Simple tensor operation on GPU to prove it works
        x = torch.rand(3, 3).cuda()
        print("GPU Sanity Check Tensor Operation: Success")
    else:
        print("\n[WARNING] CUDA is NOT available to PyTorch.")
        print("Your code will fallback to running on the CPU, which will be slow for training.")
        print("Double-check your PyTorch installation command and GPU drivers.")
    print("=" * 50)

def build_directories():
    """Programmatically creates a standardized folder structure for TwinDrive."""
    print("\nInitializing project directory structure...")
    
    # Define the directory tree
    folders = [
        # Formatted/aligned training data
        "data/processed/train/images",
        "data/processed/train/masks_drivable",
        "data/processed/train/masks_lanes",
        "data/processed/train/masks_ego",
        
        "data/processed/val/images",
        "data/processed/val/masks_drivable",
        "data/processed/val/masks_lanes",
        "data/processed/val/masks_ego",
        
        # Weights and Model outputs
        "twin/weights",
        "twin/outputs/predictions",
        
        # Source code modules
        "twin/src/models",
        "twin/src/utils",
        "twin/src/postprocess"
    ]
    
    created_count = 0
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
            created_count += 1
            
    if created_count == 0:
        print("All folders already exist! Ready for development.")
    else:
        print(f"Successfully generated {created_count} directory nodes.")
    print("=" * 50)

if __name__ == "__main__":
    check_environment()
    build_directories()
