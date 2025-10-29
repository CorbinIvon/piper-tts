# Piper TTS Voice Training Project

A complete framework for creating custom text-to-speech (TTS) voices using Piper TTS, with the ability to train models on your own voice.

## Features

- Text-to-speech synthesis with Piper TTS
- Audio data preprocessing and preparation
- Voice model training framework
- Custom voice model integration
- Comprehensive documentation for training your own voice

## Quick Start

### 1. Setup

```bash
# Activate virtual environment
. .venv/bin/activate

# Install dependencies
python3 -m pip install -r requirements.txt
```

### 2. Download a Pre-trained Voice

```bash
# Create voice_data directory and download an English voice model (bryce medium quality)
mkdir -p voice_data
cd voice_data
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/bryce/medium/en_US-bryce-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/bryce/medium/en_US-bryce-medium.onnx.json
cd ..
```

### 3. Generate Speech

```bash
python hello_world_tts.py "Hello World" output.wav voice_data/en_US-bryce-medium.onnx
```

## Train Your Own Voice

See [VOICE_TRAINING_GUIDE.md](VOICE_TRAINING_GUIDE.md) for complete instructions on:

1. Recording and preparing audio data
2. Processing audio for training
3. Training a custom voice model
4. Using your trained voice with Piper TTS

**Quick training workflow:**

```bash
# 1. Prepare your audio data
python voice_training/data_preparation.py my_recordings/ my_voice

# 2. Train the model
python voice_training/trainer.py training_data/manifest.jsonl models/my_voice

# 3. Use your custom voice
python hello_world_tts.py "Testing my voice!" output.wav models/my_voice/my_voice.onnx
```

## Project Structure

- `hello_world_tts.py` - Simple TTS script for testing
- `voice_training/` - Voice training modules
  - `data_preparation.py` - Audio preprocessing and dataset creation
  - `trainer.py` - Model training framework
- `VOICE_TRAINING_GUIDE.md` - Comprehensive training guide
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.9+
- For training: 30+ minutes of audio recordings (1+ hour recommended)
- For training: GPU recommended but not required
- For inference: CPU is sufficient

## Documentation

- [Voice Training Guide](VOICE_TRAINING_GUIDE.md) - Complete guide for training custom voices
- [Piper TTS Documentation](https://github.com/rhasspy/piper) - Official Piper TTS docs

## Technologies

- **Piper TTS** - Fast, local neural TTS
- **PyTorch** - Deep learning framework
- **Librosa** - Audio analysis
- **ONNX** - Model export format

## License

See individual package licenses for dependencies.
