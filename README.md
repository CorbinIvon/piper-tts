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
python sample_tts.py "Hello World" voice_data/en_US-bryce-medium.onnx
```

Output will be saved to `output/<timestamp>_output.wav` (e.g., `output/20251028_202901_output.wav`)

## Train Your Own Voice

Want to create a custom voice? See the [Voice Training Guide](voice_training/VOICE_TRAINING_GUIDE.md) for complete instructions on training custom voice models.

## Project Structure

- `sample_tts.py` - Simple TTS script for testing
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

- [Voice Training Guide](voice_training/VOICE_TRAINING_GUIDE.md) - Complete guide for training custom voices
- [Piper TTS Documentation](https://github.com/rhasspy/piper) - Official Piper TTS docs

## Technologies

- **Piper TTS** - Fast, local neural TTS
- **PyTorch** - Deep learning framework
- **Librosa** - Audio analysis
- **ONNX** - Model export format

## License

See individual package licenses for dependencies.
