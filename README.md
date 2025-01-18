# NetworkMonitorKokoro

## Overview
NetworkMonitorKokoro is a Flask-based service that provides advanced text-to-speech (T2S) and speech-to-text (S2T) functionalities using state-of-the-art machine learning models:

- **Text-to-Speech (T2S)**: Converts text input into high-quality synthesized speech using the Kokoro model.
- **Speech-to-Text (S2T)**: Transcribes audio files into text using OpenAI's Whisper model.

This repository leverages ONNX for efficient inference and Hugging Face's model hub for seamless model downloads.

You can see the script in action with the Free Network Monitor Assistant at [https://freenetworkmonitor.click](https://freenetworkmonitor.click).

---

## Features

- **T2S (Text-to-Speech)**
  - High-quality voice synthesis using the Kokoro ONNX model.
  - Configurable voice styles via preloaded voicepacks.

- **S2T (Speech-to-Text)**
  - Accurate audio transcription with OpenAI Whisper.
  - Handles a wide range of audio inputs.

- **Automatic Model Management**
  - Models are automatically downloaded from the Hugging Face Hub if not present locally.

- **Flask API Endpoints**
  - `/generate_audio`: Convert text into speech.
  - `/transcribe_audio`: Transcribe audio into text.

---

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.8+ (Check with `python3 --version`)
- Pip (Check with `pip --version`)
- A CUDA-enabled GPU (optional, for faster inference)

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/NetworkMonitorKokoro.git
   cd NetworkMonitorKokoro
   ```

2. **Create and activate a virtual environment**:
   - **On Linux/macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **On Windows**:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

   Once activated, you should see `(venv)` at the start of your command prompt, indicating the virtual environment is active.

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the models**:
   - The Kokoro T2S model and OpenAI Whisper S2T model will be downloaded automatically during runtime.

5. **Start the Flask server**:
   ```bash
   python app.py
   ```

6. **Deactivate the virtual environment** (optional):
   ```bash
   deactivate
   ```

---

## Running as a Linux Service

To run **NetworkMonitorKokoro** as a systemd service on Linux, follow these steps:

1. **Download the repository**:
   Clone the repository to a directory where you want to run the service:
   ```bash
   git clone https://github.com/yourusername/NetworkMonitorKokoro.git
   cd NetworkMonitorKokoro
   ```

2. **Create and activate a virtual environment**:
   Create the virtual environment inside the directory:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   Install the required Python packages in the virtual environment:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a systemd service file**:
   Open a text editor to create the service file:
   ```bash
   sudo nano /etc/systemd/system/networkmonitor-kokoro.service
   ```

   Add the following content:
   ```ini
   [Unit]
   Description=NetworkMonitorKokoro Service
   After=network.target

   [Service]
   User=yourusername
   WorkingDirectory=/path/to/NetworkMonitorKokoro
   ExecStart=/path/to/NetworkMonitorKokoro/venv/bin/python3 /path/to/NetworkMonitorKokoro/app.py
   Restart=always
   Environment=PYTHONUNBUFFERED=1

   [Install]
   WantedBy=multi-user.target
   ```

   Replace `/path/to/NetworkMonitorKokoro` with the full path to the directory where the repository was cloned and the virtual environment was created. Replace `yourusername` with your Linux username.

5. **Reload systemd**:
   Reload the systemd configuration to recognize the new service:
   ```bash
   sudo systemctl daemon-reload
   ```

6. **Start the service**:
   Start the service:
   ```bash
   sudo systemctl start networkmonitor-kokoro
   ```

7. **Enable the service to start on boot**:
   Enable the service so it starts automatically when the system boots:
   ```bash
   sudo systemctl enable networkmonitor-kokoro
   ```

8. **Check the service status**:
   Check the status of the service to ensure it’s running correctly:
   ```bash
   sudo systemctl status networkmonitor-kokoro
   ```

---

### Important Notes

- The virtual environment (`venv`) must reside inside the `NetworkMonitorKokoro` directory to keep the service self-contained.
- The `WorkingDirectory` in the service file ensures that the Flask application runs from the correct location and uses the virtual environment.
---

## Usage

### API Endpoints

#### 1. **Generate Audio**
- **Endpoint**: `/generate_audio`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "text": "Your text here",
    "output": "output_audio.wav" // Optional
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "output_path": "output_audio.wav"
  }
  ```

#### 2. **Transcribe Audio**
- **Endpoint**: `/transcribe_audio`
- **Method**: `POST`
- **Request Body**:
  - A form-data request with an audio file.
- **Response**:
  ```json
  {
    "status": "success",
    "transcription": "Your transcription here"
  }
  ```

### Example Requests

#### Generate Audio
```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, world!"}' \
     http://localhost:5000/generate_audio
```

#### Transcribe Audio
```bash
curl -X POST \
     -F "file=@path/to/your/audio/file.wav" \
     http://localhost:5000/transcribe_audio
```

---


## Dependencies

- [Transformers](https://huggingface.co/transformers)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Librosa](https://librosa.org/)
- [SoundFile](https://pysoundfile.readthedocs.io/)
- [Flask](https://flask.palletsprojects.com/)
- [Flask-CORS](https://flask-cors.readthedocs.io/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/)

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature/bugfix.
3. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- Hugging Face for providing pre-trained models.
- OpenAI for the Whisper model.
- ONNX for efficient inference.

---

## Contact

For questions or support, please open an issue or contact [support@mahadeva.co.uk](mailto:support@mahadeva.co.uk).

