# Voice Training Guide

This guide explains how to train custom voice models for Piper TTS.

## Prerequisites

Before starting, ensure:
- Project is set up (see main README.md)
- Virtual environment is activated: `. .venv/bin/activate`
- You have audio recordings ready (at least 30 minutes recommended, 1+ hour ideal)
- GPU is available for training (highly recommended but not required)

## Training Workflow Overview

1. Prepare audio recordings
2. Process audio data
3. Train voice model
4. Export to ONNX format
5. Use your custom voice

## Step 1: Prepare Your Audio Data

Collect audio recordings of your voice. For best results:

- **Quantity**: At least 30 minutes of speech, ideally 1-3 hours
- **Quality**: Clear audio with minimal background noise
- **Format**: Any format (.wav, .mp3, .flac, .m4a, .ogg)
- **Content**: Varied sentences covering different phonemes
- **Consistency**: Same microphone and environment throughout

**Recommended structure:**
```
my_voice_recordings/
├── recording_001.wav
├── recording_002.wav
├── recording_003.wav
└── ...
```

**Tips for recording:**
- Read books, articles, or podcasts aloud
- Speak naturally at your normal pace
- Avoid long pauses between sentences
- Record in a quiet environment
- Use a good quality microphone

### Step 2: Process Your Audio Data

The data preparation script will:
- Normalize audio levels
- Trim silence
- Split long recordings into manageable chunks
- Resample to consistent sample rate (22kHz)

```bash
python voice_training/data_preparation.py my_voice_recordings/ my_voice
```

**Arguments:**
- `my_voice_recordings/`: Directory containing your audio files
- `my_voice`: Speaker ID (identifier for your voice)

**Output:**
```
training_data/
├── audio/
│   ├── my_voice_recording_001_000.wav
│   ├── my_voice_recording_001_001.wav
│   └── ...
└── manifest.jsonl
```

The script will display statistics about your processed dataset:
```json
{
  "total_files": 150,
  "total_duration_minutes": 45.2,
  "average_duration": 5.3,
  "min_duration": 1.2,
  "max_duration": 9.8,
  "num_speakers": 1,
  "speakers": ["my_voice"]
}
```

## Step 3 (Optional): Add Transcriptions

For better training results, provide text transcriptions of your audio.

Create a JSON file `transcriptions.json`:
```json
{
  "recording_001": "This is the text spoken in recording 001",
  "recording_002": "This is the text spoken in recording 002",
  "recording_003": "Another example transcription"
}
```

Then run with transcriptions:
```bash
python voice_training/data_preparation.py my_voice_recordings/ my_voice --transcriptions transcriptions.json
```

## Step 4: Train Your Voice Model

```bash
python voice_training/trainer.py training_data/manifest.jsonl models/my_voice
```

**Training parameters you can adjust in `trainer.py`:**
- `batch_size`: Number of samples per batch (default: 8)
- `learning_rate`: Learning rate for optimizer (default: 1e-4)
- `num_epochs`: Number of training epochs (default: 50)

**Training will:**
- Process your audio into mel spectrograms
- Train a neural TTS model
- Save checkpoints every 10 epochs
- Log training metrics to TensorBoard

**Monitor training progress:**
```bash
tensorboard --logdir models/my_voice/logs
```

Open http://localhost:6006 in your browser to view training metrics.

## Step 5: Export to ONNX

Once training is complete, export your model to ONNX format for use with Piper:

```bash
python voice_training/trainer.py export models/my_voice/checkpoint_epoch_50.pt models/my_voice/my_voice.onnx
```

## Using Your Custom Voice

Once you have your trained model, use it just like any Piper voice:

```bash
python hello_world_tts.py "This is my custom voice!" output.wav models/my_voice/my_voice.onnx
```

## Advanced: Training Considerations

### Model Architecture

The current framework provides the structure for training. To complete it, you need to:

1. **Choose a TTS architecture**:
   - Tacotron 2: Classic seq2seq model
   - FastSpeech 2: Non-autoregressive, faster training
   - VITS: End-to-end model with vocoder

2. **Implement the model in `trainer.py`**:
   - Replace the placeholder model initialization
   - Add proper loss functions
   - Implement the forward pass

3. **Use pre-trained Piper model as base**:
   - Fine-tune an existing Piper model on your voice
   - Requires less data and training time
   - Better results with limited data

### Data Quality vs Quantity

| Data Amount | Expected Quality |
|-------------|------------------|
| 30 min      | Recognizable but robotic |
| 1 hour      | Good quality, some artifacts |
| 3+ hours    | High quality, natural sounding |

### Hardware Requirements

**Training:**
- **CPU only**: Possible but very slow (days/weeks)
- **GPU**: Recommended, significantly faster (hours/days)
  - Minimum: 6GB VRAM
  - Recommended: 8GB+ VRAM

**Inference:**
- CPU is sufficient for real-time synthesis
- Piper is optimized for CPU inference

## Troubleshooting

### Installation Issues

If you encounter issues with `piper-tts`:
```bash
pip install --upgrade piper-tts
```

### Audio Quality Issues

- **Robotic sound**: Need more training data
- **Artifacts/glitches**: Lower learning rate or train longer
- **Wrong pronunciation**: Add more transcriptions
- **Inconsistent voice**: Ensure consistent recording environment

### Training Issues

- **Out of memory**: Reduce batch size
- **Training too slow**: Check if GPU is being used
- **Model not improving**: Adjust learning rate or check data quality

## File Structure

```
piper-tts/
├── hello_world_tts.py          # Basic TTS script
├── voice_training/
│   ├── data_preparation.py     # Audio preprocessing
│   └── trainer.py              # Model training
├── training_data/              # Processed training data
│   ├── audio/                  # Processed audio clips
│   └── manifest.jsonl          # Dataset manifest
├── models/                     # Trained models
│   └── my_voice/
│       ├── checkpoint_*.pt     # Training checkpoints
│       ├── my_voice.onnx       # Exported ONNX model
│       └── logs/               # TensorBoard logs
└── requirements.txt            # Python dependencies
```

## Next Steps

1. Experiment with different voices and speaking styles
2. Fine-tune training parameters for better quality
3. Create a voice dataset with diverse content
4. Integrate with applications that need TTS

## Resources

- [Piper TTS GitHub](https://github.com/rhasspy/piper)
- [Piper Voice Models](https://github.com/rhasspy/piper/releases)
- [TTS Research Papers](https://paperswithcode.com/task/text-to-speech-synthesis)
- [Audio Processing with Librosa](https://librosa.org/doc/latest/index.html)

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the Piper TTS documentation
- Examine training logs and error messages
