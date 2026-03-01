import platform
import subprocess
import sys


COMMON_REQUIREMENTS = [
    "flask",
    "flask-cors",
    "gunicorn",
    "transformers",
    "librosa",
    "numpy",
    "soundfile",
    "huggingface_hub",
    "phonemizer",
    "munch",
    "werkzeug",
    "num2words",
    "dateparser",
    "inflect",
    "ftfy",
    "sentencepiece",
]


def install_package(package, extra_args=None):
    command = [sys.executable, "-m", "pip", "install", package]
    if extra_args:
        command.extend(extra_args)
    subprocess.check_call(command)


def install_python_deps(choice):
    if choice == "1":
        print("\nInstalling Python dependencies for CPU...")
        for req in COMMON_REQUIREMENTS:
            print(f"Installing {req}...")
            install_package(req)
        install_package("torch", ["--index-url", "https://download.pytorch.org/whl/cpu"])
        install_package("onnxruntime")
    elif choice == "2":
        print("\nInstalling Python dependencies for GPU...")
        for req in COMMON_REQUIREMENTS:
            print(f"Installing {req}...")
            install_package(req)
        install_package("torch")
        install_package("onnxruntime-gpu")
    else:
        raise ValueError("Invalid install mode")


def main():
    print("Detecting operating system...")
    os_type = platform.system()
    print(f"Operating system detected: {os_type}")
    print("Note: privileged system installs (apt + optional Piper binary) are handled by install.sh.")

    print("\nSelect installation mode:")
    print("1. CPU (no GPU dependencies)")
    print("2. GPU (requires CUDA-compatible hardware and drivers)")
    choice = input("Enter 1 or 2: ").strip()

    try:
        install_python_deps(choice)
    except ValueError:
        print("\nInvalid choice. Exiting.")
        sys.exit(1)

    print("\nPython dependency installation complete!")


if __name__ == "__main__":
    main()
