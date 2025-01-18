import subprocess
import sys

# Define the common requirements
common_requirements = [
    "flask",
    "flask-cors",
    "transformers",
    "librosa",
    "numpy",
    "onnxruntime",
    "soundfile",
    "huggingface_hub",
    "phonemizer",
    "munch"
]

# Function to run pip install
def install_package(package, extra_args=None):
    try:
        command = [sys.executable, "-m", "pip", "install", package]
        if extra_args:
            command.extend(extra_args)
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")

# Main installation logic
def main():
    print("Select installation mode:")
    print("1. CPU (no GPU dependencies)")
    print("2. GPU (requires CUDA-compatible hardware and drivers)")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("\nInstalling for CPU...")
        # Install common dependencies
        for req in common_requirements:
            print(f"Installing {req}...")
            install_package(req)
        # Install PyTorch for CPU
        print("Installing torch (CPU-only)...")
        install_package("torch", ["--index-url", "https://download.pytorch.org/whl/cpu"])
    elif choice == "2":
        print("\nInstalling for GPU...")
        # Install common dependencies
        for req in common_requirements:
            print(f"Installing {req}...")
            install_package(req)
        # Install default PyTorch for GPU
        print("Installing torch (GPU)...")
        install_package("torch")
    else:
        print("\nInvalid choice. Exiting.")
        sys.exit(1)

    print("\nInstallation complete!")

if __name__ == "__main__":
    main()

