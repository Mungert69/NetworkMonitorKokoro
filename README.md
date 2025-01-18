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
- Python 3.8+
- Pip
- A CUDA-enabled GPU (optional, for faster inference)

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/NetworkMonitorKokoro.git
   cd NetworkMonitorKokoro
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the models:
   - The Kokoro T2S model and OpenAI Whisper S2T model will be downloaded automatically during runtime.

4. Start the Flask server:
   ```bash
   python app.py
   ```

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

## Running as a Linux Service

To run NetworkMonitorKokoro as a systemd service on Linux, follow these steps:

1. Create a service file:
   ```bash
   sudo nano /etc/systemd/system/networkmonitor-kokoro.service
   ```

2. Add the following content to the file:
   ```ini
   [Unit]
   Description=NetworkMonitorKokoro Service
   After=network.target

   [Service]
   User=yourusername
   WorkingDirectory=/path/to/NetworkMonitorKokoro
   ExecStart=/usr/bin/python3 /path/to/NetworkMonitorKokoro/app.py
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   Replace `/path/to/NetworkMonitorKokoro` with the full path to the cloned repository and `yourusername` with your Linux username.

3. Reload systemd to recognize the new service:
   ```bash
   sudo systemctl daemon-reload
   ```

4. Start the service:
   ```bash
   sudo systemctl start networkmonitor-kokoro
   ```

5. Enable the service to start on boot:
   ```bash
   sudo systemctl enable networkmonitor-kokoro
   ```

6. Check the service status:
   ```bash
   sudo systemctl status networkmonitor-kokoro
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

