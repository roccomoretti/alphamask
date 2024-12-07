#!/usr/bin/env python3
import sys
import subprocess

def get_package_version(package):
    try:
        import importlib
        return importlib.import_module(package).__version__
    except ImportError:
        return f"{package} not installed"
    except AttributeError:
        return f"Version info not available for {package}"

def main():
    # Check Python version
    print(f"Python version: {sys.version.split()[0]}")
    
    # Check JAX and JAXLIB versions
    print(f"JAX version: {get_package_version('jax')}")
    print(f"JAXLIB version: {get_package_version('jaxlib')}")
    
    # Check CUDA version if available
    try:
        nvidia_smi = subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version,cuda_version', '--format=csv,noheader']).decode()
        driver_version, cuda_version = nvidia_smi.strip().split(', ')
        print(f"NVIDIA driver version: {driver_version}")
        print(f"CUDA version: {cuda_version}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("NVIDIA driver/CUDA info not available")
    
    # Check if JAX can see GPU
    try:
        import jax
        devices = jax.devices()
        print("\nAvailable JAX devices:")
        for device in devices:
            print(f"- {device}")
    except ImportError:
        print("\nCould not check JAX devices (JAX not installed)")

if __name__ == "__main__":
    main() 