#!/usr/bin/env python3
"""
Voice model training module for Piper TTS
This module provides the framework for training custom voice models
"""

import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter
from model import VocoderModel, SimpleTTSModel


class VoiceDataset(Dataset):
    """PyTorch Dataset for voice training data"""

    def __init__(self, manifest_path: str, sample_rate: int = 22050):
        """
        Initialize the dataset

        Args:
            manifest_path: Path to the manifest.jsonl file
            sample_rate: Target sample rate
        """
        self.sample_rate = sample_rate
        self.data = []

        # Load manifest
        with open(manifest_path, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

        print(f"Loaded {len(self.data)} audio samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single training sample"""
        item = self.data[idx]

        # Load audio using soundfile directly to avoid torchcodec issues
        audio_data, sr = sf.read(item["audio_path"])
        # Convert to tensor and add channel dimension if mono
        waveform = torch.from_numpy(audio_data).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
        else:
            waveform = waveform.t()  # Transpose to (channels, samples)

        # Ensure correct sample rate
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Extract features (mel spectrogram)
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
        )
        mel_spec = mel_transform(waveform)

        return {
            "mel_spec": mel_spec.squeeze(0),
            "audio": waveform.squeeze(0),
            "text": item.get("text", ""),
            "speaker_id": item["speaker_id"],
        }


class VoiceTrainer:
    """Trainer for voice models"""

    def __init__(
        self,
        manifest_path: str,
        output_dir: str = "models",
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
    ):
        """
        Initialize the trainer

        Args:
            manifest_path: Path to the training manifest
            output_dir: Directory to save trained models
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
            num_epochs: Number of training epochs
        """
        self.manifest_path = manifest_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create dataset and dataloader
        self.dataset = VoiceDataset(manifest_path)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate_fn,
        )

        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))

    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        # Find max length in batch
        max_len = max(item["mel_spec"].shape[1] for item in batch)

        # Pad sequences
        mel_specs = []
        audios = []
        texts = []
        speaker_ids = []

        for item in batch:
            mel = item["mel_spec"]
            pad_len = max_len - mel.shape[1]
            if pad_len > 0:
                mel = torch.nn.functional.pad(mel, (0, pad_len))
            mel_specs.append(mel)
            audios.append(item["audio"])
            texts.append(item["text"])
            speaker_ids.append(item["speaker_id"])

        return {
            "mel_specs": torch.stack(mel_specs),
            "audios": audios,
            "texts": texts,
            "speaker_ids": speaker_ids,
        }

    def train(self):
        """
        Train the voice model using a vocoder-based approach
        """
        print("Starting training...")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Dataset size: {len(self.dataset)}")

        # Initialize vocoder model for voice characteristic learning
        model = VocoderModel(n_mels=80, hidden_dim=256).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        for epoch in range(self.num_epochs):
            model.train()
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Training loop
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(self.dataloader):
                mel_specs = batch["mel_specs"].to(self.device)

                # Forward pass: Reconstruct mel spectrograms
                optimizer.zero_grad()
                reconstructed = model(mel_specs)

                # Calculate reconstruction loss
                loss = criterion(reconstructed, mel_specs)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                if (batch_idx + 1) % 5 == 0:
                    print(
                        f"  Batch {batch_idx + 1}/{len(self.dataloader)}, Loss: {loss.item():.4f}"
                    )

            # Log epoch metrics
            avg_loss = epoch_loss / len(self.dataloader)
            self.writer.add_scalar("Loss/train", avg_loss, epoch)
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

        # Save final model
        final_model_path = self.output_dir / "final_model.pt"
        torch.save(model.state_dict(), final_model_path)
        print(f"\nFinal model saved: {final_model_path}")

        print("\nTraining completed!")
        self.writer.close()

        return model

    def export_onnx(self, model_path: str, output_path: str):
        """
        Export trained model to ONNX format

        Args:
            model_path: Path to the trained PyTorch model
            output_path: Output path for ONNX model
        """
        print(f"Exporting model to ONNX: {output_path}")

        # Load the model
        model = VocoderModel(n_mels=80, hidden_dim=256).to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        # Create dummy input (batch_size=1, n_mels=80, time_steps=100)
        dummy_input = torch.randn(1, 80, 100).to(self.device)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['mel_input'],
            output_names=['mel_output'],
            dynamic_axes={
                'mel_input': {0: 'batch_size', 2: 'time'},
                'mel_output': {0: 'batch_size', 2: 'time'}
            }
        )

        print(f"ONNX export completed: {output_path}")

        # Also create a config file for Piper compatibility
        config_path = output_path + ".json"
        config = {
            "audio": {
                "sample_rate": 22050,
                "quality": "medium"
            },
            "espeak": {
                "voice": "en-us"
            },
            "inference": {
                "noise_scale": 0.667,
                "length_scale": 1.0,
                "noise_w": 0.8
            },
            "phoneme_type": "text",
            "n_speakers": 1,
            "speaker_id_map": {
                "glados": 0
            }
        }

        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Config file created: {config_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python trainer.py <manifest_path> [output_dir]")
        print("\nExample:")
        print("  python trainer.py training_data/manifest.jsonl models/my_voice")
        sys.exit(1)

    manifest_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "models"

    # Create trainer and start training
    trainer = VoiceTrainer(
        manifest_path=manifest_path,
        output_dir=output_dir,
        batch_size=8,
        learning_rate=1e-4,
        num_epochs=50,
    )

    trainer.train()
