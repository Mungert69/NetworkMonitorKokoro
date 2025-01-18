from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Step 1: Load the model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Disable forced language or task specification
model.config.forced_decoder_ids = None

# Step 2: Load your audio file
audio_path = "/home/mahadeva/code/securefiles/dev/output_audio/output_audio_am_adam_ffdd95e1-f907-4cb9-8c75-df4155f402fc.wav"  # Replace with your file path
audio_array, sampling_rate = librosa.load(audio_path, sr=16000)  # Load and resample to 16kHz

# Step 3: Process audio input
input_features = processor(
    audio_array, 
    sampling_rate=sampling_rate, 
    return_tensors="pt"
).input_features

# Step 4: Generate predictions (token IDs)
predicted_ids = model.generate(input_features)

# Step 5: Decode token IDs to text
# Include special tokens
transcription_with_tokens = processor.batch_decode(predicted_ids, skip_special_tokens=False)
print("Transcription with special tokens:")
print(transcription_with_tokens)

# Exclude special tokens
transcription_without_tokens = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print("\nTranscription without special tokens:")
print(transcription_without_tokens)

