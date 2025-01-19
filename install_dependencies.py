import subprocess
import sys
import platform

# Define the common requirements
common_requirements = [
    "flask",
    "flask-cors",
    "transformers",
    "librosa",
    "numpy",
    "soundfile",
    "huggingface_hub",
    "phonemizer",
    "munch",
    "werkzeug",
    "tempfile",
    "num2words",
    "dateparser",
    "hashlib"
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
    print("Detecting operating system...")
    os_type = platform.system()
    print(f"Operating system detected: {os_type}")

    print("\nSelect installation mode:")
    print("1. CPU (no GPU dependencies)")
    print("2. GPU (requires CUDA-compatible hardware and drivers)")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        print("\nInstalling for CPU...")
        for req in common_requirements:
            print(f"Installing {req}...")
            install_package(req)
        print("Installing torch (CPU-only)...")
        install_package("torch", ["--index-url", "https://download.pytorch.org/whl/cpu"])
        print("Installing onnxruntime (CPU-only)...")
        install_package("onnxruntime")
    elif choice == "2":
        print("\nInstalling for GPU...")
        for req in common_requirements:
            print(f"Installing {req}...")
            install_package(req)
        print("Installing torch (GPU)...")
        install_package("torch")
        print("Installing onnxruntime-gpu...")
        install_package("onnxruntime-gpu")
    else:
        print("\nInvalid choice. Exiting.")
        sys.exit(1)

    print("\nInstallation complete!")

if __name__ == "__main__":
    main()

