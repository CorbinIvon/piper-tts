#!/usr/bin/env python3
"""
Basic TTS script using Piper TTS to synthesize 'Hello World'
"""

import sys
import wave
from pathlib import Path
from datetime import datetime
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
        print("Example: python hello_world_tts.py 'Hello World' path/to/model.onnx")
        print("Output will be saved to: output/<timestamp>_output.wav")
        sys.exit(1)

    voice_model_path = Path(voice_model)
    if not voice_model_path.exists():
        print(f"Error: Voice model not found at {voice_model}")
        sys.exit(1)

    # Load the voice model
    print(f"Loading voice model: {voice_model}")
    voice = PiperVoice.load(voice_model)

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Synthesize speech
    print(f"Synthesizing: '{text}'")

    with wave.open(str(output_path), 'wb') as wav_file:
        # Synthesize returns an iterable of AudioChunk objects
        audio_chunks = list(voice.synthesize(text))

        if not audio_chunks:
            print("Error: No audio generated")
            sys.exit(1)

        # Configure wave file with parameters from first chunk
        first_chunk = audio_chunks[0]
        wav_file.setnchannels(first_chunk.sample_channels)
        wav_file.setsampwidth(first_chunk.sample_width)
        wav_file.setframerate(first_chunk.sample_rate)

        # Write all audio chunks
        for chunk in audio_chunks:
            # Get audio as int16 bytes
            if hasattr(chunk, 'audio_int16_bytes'):
                wav_file.writeframes(chunk.audio_int16_bytes)
            else:
                # Fallback: convert float array to int16
                import numpy as np
                audio_int16 = (chunk.audio_float_array * 32767).astype(np.int16)
                wav_file.writeframes(audio_int16.tobytes())

    print(f"Audio saved to: {output_file}")


if __name__ == "__main__":
    # Generate timestamped output filename (always used)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = f"output/{timestamp}_output.wav"

    if len(sys.argv) < 2:
        # No arguments: use defaults
        text = "Hello World"
        model = None
    elif len(sys.argv) == 2:
        # One argument: text only
        text = sys.argv[1]
        model = None
    else:
        # Two+ arguments: text and model
        text = sys.argv[1]
        model = sys.argv[2]

    text_to_speech(text, output, model)
