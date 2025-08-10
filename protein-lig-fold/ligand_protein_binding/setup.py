#!/usr/bin/env python3

import subprocess
import sys
import os

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    # Install PyTorch with CUDA support
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    
    # Install PyTorch Geometric
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "torch-geometric", 
        "-f", "https://data.pyg.org/whl/torch-2.0.0+cu118.html"
    ])
    
    # Install other requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # Install RDKit via conda if available, otherwise pip
    try:
        subprocess.check_call(["conda", "install", "-c", "rdkit", "rdkit", "-y"])
    except FileNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit"])
    
    print("✅ Dependencies installed successfully!")

def setup_authentication():
    """Setup authentication for Weights & Biases and Hugging Face"""
    print("\n🔐 Setting up authentication...")
    
    print("Please run the following commands to authenticate:")
    print("1. For Weights & Biases: wandb login")
    print("2. For Hugging Face: huggingface-cli login")
    
    # Prompt for authentication
    input("Press Enter after completing authentication...")

def main():
    install_dependencies()
    setup_authentication()
    print("\n🎉 Setup completed! You can now run: python run.py")

if __name__ == "__main__":
    main()
