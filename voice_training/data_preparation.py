#!/usr/bin/env python3
"""
Data preparation module for voice training
Handles audio file ingestion, preprocessing, and dataset creation
"""

import os
import json
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from pydub import AudioSegment


class VoiceDataPreparator:
    """Prepare audio data for voice model training"""

    def __init__(self, output_dir: str = "training_data"):
        """
        Initialize the data preparator

        Args:
            output_dir: Directory to store processed training data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(exist_ok=True)

        # Target audio parameters for training
        self.sample_rate = 22050
        self.target_duration = 10  # max duration in seconds

    def process_audio_file(
        self, audio_path: str, text: str = None, speaker_id: str = "default"
    ) -> Dict:
        """
        Process a single audio file for training

        Args:
            audio_path: Path to the audio file
            text: Transcription of the audio (if available)
            speaker_id: Identifier for the speaker

        Returns:
            Dictionary with processed audio info
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"Processing: {audio_path.name}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Normalize audio
        audio = librosa.util.normalize(audio)

        # Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=20)

        # Split long audio into chunks if needed
        duration = len(audio) / self.sample_rate
        chunks = []

        if duration > self.target_duration:
            # Split into chunks
            chunk_samples = int(self.target_duration * self.sample_rate)
            for i in range(0, len(audio), chunk_samples):
                chunk = audio[i : i + chunk_samples]
                if len(chunk) > self.sample_rate:  # At least 1 second
                    chunks.append(chunk)
        else:
            chunks = [audio]

        # Save processed chunks
        results = []
        for idx, chunk in enumerate(chunks):
            output_filename = f"{speaker_id}_{audio_path.stem}_{idx:03d}.wav"
            output_path = self.audio_dir / output_filename

            sf.write(output_path, chunk, self.sample_rate)

            results.append(
                {
                    "audio_path": str(output_path),
                    "speaker_id": speaker_id,
                    "duration": len(chunk) / self.sample_rate,
                    "text": text if text else "",
                    "sample_rate": self.sample_rate,
                }
            )

        return results

    def process_directory(
        self, audio_dir: str, transcription_file: str = None, speaker_id: str = "default"
    ) -> List[Dict]:
        """
        Process all audio files in a directory

        Args:
            audio_dir: Directory containing audio files
            transcription_file: Optional JSON file with transcriptions
            speaker_id: Identifier for the speaker

        Returns:
            List of processed audio info dictionaries
        """
        audio_dir = Path(audio_dir)
        if not audio_dir.exists():
            raise FileNotFoundError(f"Directory not found: {audio_dir}")

        # Load transcriptions if provided
        transcriptions = {}
        if transcription_file:
            with open(transcription_file, "r") as f:
                transcriptions = json.load(f)

        # Process all audio files
        all_results = []
        audio_extensions = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

        for audio_file in audio_dir.iterdir():
            if audio_file.suffix.lower() in audio_extensions:
                text = transcriptions.get(audio_file.stem, "")
                try:
                    results = self.process_audio_file(
                        str(audio_file), text=text, speaker_id=speaker_id
                    )
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error processing {audio_file.name}: {e}")

        return all_results

    def create_dataset_manifest(
        self, audio_info_list: List[Dict], output_file: str = "manifest.jsonl"
    ):
        """
        Create a manifest file for the training dataset

        Args:
            audio_info_list: List of processed audio information
            output_file: Output manifest file name
        """
        manifest_path = self.output_dir / output_file

        with open(manifest_path, "w") as f:
            for info in audio_info_list:
                f.write(json.dumps(info) + "\n")

        print(f"Dataset manifest created: {manifest_path}")
        print(f"Total audio files: {len(audio_info_list)}")
        total_duration = sum(info["duration"] for info in audio_info_list)
        print(f"Total duration: {total_duration / 60:.2f} minutes")

    def get_audio_stats(self, audio_info_list: List[Dict]) -> Dict:
        """
        Get statistics about the processed audio dataset

        Args:
            audio_info_list: List of processed audio information

        Returns:
            Dictionary with dataset statistics
        """
        durations = [info["duration"] for info in audio_info_list]
        speakers = set(info["speaker_id"] for info in audio_info_list)

        stats = {
            "total_files": len(audio_info_list),
            "total_duration_seconds": sum(durations),
            "total_duration_minutes": sum(durations) / 60,
            "average_duration": np.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "num_speakers": len(speakers),
            "speakers": list(speakers),
        }

        return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_preparation.py <audio_directory> [speaker_id]")
        print("\nExample:")
        print("  python data_preparation.py ./my_voice_samples my_voice")
        sys.exit(1)

    audio_dir = sys.argv[1]
    speaker_id = sys.argv[2] if len(sys.argv) > 2 else "default"

    # Process the audio directory
    preparator = VoiceDataPreparator()
    results = preparator.process_directory(audio_dir, speaker_id=speaker_id)

    # Create manifest
    preparator.create_dataset_manifest(results)

    # Print statistics
    stats = preparator.get_audio_stats(results)
    print("\nDataset Statistics:")
    print(json.dumps(stats, indent=2))
