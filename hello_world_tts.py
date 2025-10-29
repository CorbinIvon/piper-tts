#!/usr/bin/env python3
"""
Basic TTS script using Piper TTS to synthesize 'Hello World'
"""

import sys
import wave
from pathlib import Path
from piper import PiperVoice


def text_to_speech(text: str, output_file: str = "output.wav", voice_model: str = None):
    """
    Convert text to speech using Piper TTS

    Args:
        text: Text to convert to speech
        output_file: Output WAV file path
        voice_model: Path to the voice model file (.onnx)
    """
    if voice_model is None:
        print("Error: Voice model path required.")
        print("Download a model from: https://github.com/rhasspy/piper/releases/tag/v1.0.0")
        print("Example: python hello_world_tts.py 'Hello World' output.wav path/to/model.onnx")
        sys.exit(1)

    voice_model_path = Path(voice_model)
    if not voice_model_path.exists():
        print(f"Error: Voice model not found at {voice_model}")
        sys.exit(1)

    # Load the voice model
    print(f"Loading voice model: {voice_model}")
    voice = PiperVoice.load(voice_model)

    # Synthesize speech
    print(f"Synthesizing: '{text}'")
    with wave.open(output_file, 'wb') as wav_file:
        voice.synthesize(text, wav_file)

    print(f"Audio saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        text = "Hello World"
        output = "output.wav"
        model = None
    elif len(sys.argv) < 3:
        text = sys.argv[1]
        output = "output.wav"
        model = None
    elif len(sys.argv) < 4:
        text = sys.argv[1]
        output = sys.argv[2]
        model = None
    else:
        text = sys.argv[1]
        output = sys.argv[2]
        model = sys.argv[3]

    text_to_speech(text, output, model)
