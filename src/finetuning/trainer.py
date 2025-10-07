"""Client-specific CLIP fine-tuning with LoRA adapters."""

import os
from pathlib import Path
from typing import List, Dict, Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

from transformers import CLIPModel, CLIPProcessor
from peft import LoraConfig, get_peft_model, TaskType

import logging

LOGGER = logging.getLogger(__name__)


class VideoTextDataset(Dataset):
    """Dataset for video-text pairs."""
    
    def __init__(self, video_paths: List[Path], labels: List[List[str]], processor):
        self.video_paths = video_paths
        self.labels = labels
        self.processor = processor
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        labels = self.labels[idx]
        
        # Load video frames (simplified - use middle frame)
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        
        # Get middle frame
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Fallback to black frame
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use first label only to avoid variable length issues
        text = labels[0] if isinstance(labels, list) else labels
        
        return {
            'frame': frame,
            'text': text
        }


class ClientFineTuner:
    """Fine-tune CLIP for client-specific videos using LoRA."""
    
    def __init__(
        self,
        base_model: str = "openai/clip-vit-base-patch32",
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        device: str = "cuda"
    ):
        """
        Initialize fine-tuner.
        
        Args:
            base_model: Base CLIP model name
            lora_rank: LoRA rank (lower = faster, higher = more capacity)
            lora_alpha: LoRA alpha (scaling factor)
            lora_dropout: Dropout rate
            device: Device to train on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.base_model_name = base_model
        
        LOGGER.info(f"Loading base model: {base_model}")
        
        # Load base model
        self.model = CLIPModel.from_pretrained(base_model)
        self.processor = CLIPProcessor.from_pretrained(base_model)
        
        # Configure LoRA - use regex pattern for all attention layers
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=r".*\.(q_proj|v_proj)$",  # All q_proj and v_proj layers (regex)
            lora_dropout=lora_dropout,
            bias="none",
            modules_to_save=None
        )
        
        # Add LoRA adapters
        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        LOGGER.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def prepare_dataset(
        self,
        video_dir: Path,
        labels_file: Optional[Path] = None
    ) -> Dataset:
        """
        Prepare dataset from client videos.
        
        Args:
            video_dir: Directory containing client videos
            labels_file: JSON file with video labels (optional)
        
        Returns:
            PyTorch Dataset
        """
        video_paths = []
        labels = []
        
        if labels_file and labels_file.exists():
            # Load labels from file
            with open(labels_file) as f:
                labels_data = json.load(f)
            
            for item in labels_data:
                video_path = Path(item['video_path'])
                if video_path.exists():
                    video_paths.append(video_path)
                    labels.append(item['labels'])
        else:
            # Auto-generate labels from folder structure
            for video_path in video_dir.rglob("*.mp4"):
                video_paths.append(video_path)
                # Use parent folder name as label
                label = video_path.parent.name
                labels.append([label, f"video of {label}", f"{label} action"])
        
        LOGGER.info(f"Prepared dataset with {len(video_paths)} videos")
        
        return VideoTextDataset(video_paths, labels, self.processor)
    
    def contrastive_loss(self, image_embeds, text_embeds, temperature=0.07):
        """Compute contrastive loss (CLIP-style)."""
        # Normalize embeddings
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_embeds, text_embeds.t()) / temperature
        
        # Labels: diagonal elements are positive pairs
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Symmetric loss (image-to-text and text-to-image)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def train(
        self,
        dataset: Dataset,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        output_dir: Path = Path("models/finetuned"),
        client_name: str = "client"
    ):
        """
        Fine-tune model on client dataset.
        
        Args:
            dataset: Training dataset
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            output_dir: Directory to save model
            client_name: Client identifier
        """
        LOGGER.info(f"Starting fine-tuning for client: {client_name}")
        LOGGER.info(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress_bar:
                # Process batch
                frames = batch['frame']
                texts = batch['text']
                
                # Process images and text
                image_inputs = self.processor(images=frames, return_tensors="pt")
                text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
                
                # Move to device
                pixel_values = image_inputs['pixel_values'].to(self.device)
                input_ids = text_inputs['input_ids'].to(self.device)
                attention_mask = text_inputs['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get embeddings
                image_embeds = outputs.image_embeds
                text_embeds = outputs.text_embeds
                
                # Compute loss
                loss = self.contrastive_loss(image_embeds, text_embeds)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = epoch_loss / len(dataloader)
            LOGGER.info(f"Epoch {epoch+1}/{epochs} - Average loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_model(output_dir, client_name)
                LOGGER.info(f"Saved best model (loss: {best_loss:.4f})")
        
        LOGGER.info(f"Fine-tuning complete! Best loss: {best_loss:.4f}")
        
        return best_loss
    
    def save_model(self, output_dir: Path, client_name: str):
        """Save fine-tuned model."""
        output_dir = output_dir / client_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapters only (lightweight)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)
        
        # Save metadata
        metadata = {
            'client_name': client_name,
            'base_model': self.base_model_name,
            'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        LOGGER.info(f"Model saved to {output_dir}")
    
    @classmethod
    def load_finetuned(cls, model_dir: Path, device: str = "cuda"):
        """Load fine-tuned model."""
        from peft import PeftModel
        
        # Load metadata
        with open(model_dir / 'metadata.json') as f:
            metadata = json.load(f)
        
        # Load base model
        base_model = CLIPModel.from_pretrained(metadata['base_model'])
        
        # Load LoRA adapters
        model = PeftModel.from_pretrained(base_model, model_dir)
        processor = CLIPProcessor.from_pretrained(model_dir)
        
        model.to(device)
        model.eval()
        
        LOGGER.info(f"Loaded fine-tuned model for client: {metadata['client_name']}")
        
        return model, processor


__all__ = ["ClientFineTuner", "VideoTextDataset"]
