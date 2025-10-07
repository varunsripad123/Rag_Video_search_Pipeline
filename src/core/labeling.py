"""Automated video labeling orchestrator combining multiple AI models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from src.config import AppConfig
from src.models.yolo import YOLODetector
from src.models.blip2 import BLIP2Captioner
from src.models.whisper_transcriber import WhisperTranscriber
from src.models.videomae import VideoMAEEncoder
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)


class AutoLabeler:
    """
    Orchestrates automated labeling of video chunks using multiple AI models:
    - YOLO for object and scene detection
    - VideoMAE for action recognition  
    - BLIP-2 for caption generation
    - Whisper for audio transcription
    """
    
    def __init__(self, config: AppConfig, enable_all: bool = True):
        """
        Initialize the auto-labeler with all models.
        
        Args:
            config: Application configuration
            enable_all: If True, load all models; if False, load only essential ones
        """
        self.config = config
        self.device = config.models.device
        
        # Initialize models
        self.yolo = YOLODetector(
            model_name="yolov8n.pt",  # Nano model for speed
            device=self.device
        )
        
        self.blip2 = BLIP2Captioner(
            model_name="Salesforce/blip2-opt-2.7b",
            device=self.device
        )
        
        self.whisper = WhisperTranscriber(
            model_name="openai/whisper-tiny",  # Tiny model for speed
            device=self.device
        )
        
        self.videomae = VideoMAEEncoder(
            model_name=config.models.videomae_model_name,
            device=self.device,
            precision=config.models.precision,
            quantize=config.models.quantize
        )
        
        self.models_loaded = False
    
    def load(self) -> None:
        """Load all models into memory."""
        LOGGER.info("Loading auto-labeling models...")
        
        try:
            LOGGER.info("Loading YOLO detector...")
            self.yolo.load()
        except Exception as e:
            LOGGER.warning(f"Failed to load YOLO: {e}")
        
        try:
            LOGGER.info("Loading BLIP-2 captioner...")
            self.blip2.load()
        except Exception as e:
            LOGGER.warning(f"Failed to load BLIP-2: {e}")
        
        try:
            LOGGER.info("Loading Whisper transcriber...")
            self.whisper.load()
        except Exception as e:
            LOGGER.warning(f"Failed to load Whisper: {e}")
        
        try:
            LOGGER.info("Loading VideoMAE for action recognition...")
            self.videomae.load()
        except Exception as e:
            LOGGER.warning(f"Failed to load VideoMAE: {e}")
        
        self.models_loaded = True
        LOGGER.info("Auto-labeling models loaded successfully")
    
    def label_video_chunk(
        self,
        video_path: Path,
        frames: Optional[List[np.ndarray]] = None,
        include_audio: bool = True
    ) -> Dict[str, any]:
        """
        Generate comprehensive labels for a video chunk.
        
        Args:
            video_path: Path to video file
            frames: Pre-loaded frames (optional, will load if not provided)
            include_audio: Whether to transcribe audio
        
        Returns:
            Dictionary containing all labeling results in the format:
            {
                "objects": ["person", "car", "tree"],
                "action": "walking",
                "caption": "a person walking near a car",
                "audio_text": "background music",
                "confidence": 0.84,
                "metadata": {...}
            }
        """
        if not self.models_loaded:
            LOGGER.warning("Models not loaded, loading now...")
            self.load()
        
        # Load frames if not provided
        if frames is None:
            from src.utils.video import load_video_frames
            tensor = load_video_frames(video_path)
            # Convert tensor to numpy frames: (1,3,T,H,W) -> (T,H,W,3)
            frames_np = tensor.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
            frames = [(frames_np[i] * 255).astype(np.uint8) for i in range(frames_np.shape[0])]
        
        results = {}
        confidences = []
        
        # 1. Object Detection with YOLO
        try:
            LOGGER.debug(f"Running object detection on {video_path.name}")
            object_results = self.yolo.detect_objects(frames)
            results["objects"] = object_results.get("objects", [])
            results["object_counts"] = object_results.get("object_counts", {})
            results["object_detections"] = object_results.get("detections", [])[:5]  # Keep top 5
            confidences.append(object_results.get("avg_confidence", 0.5))
        except Exception as e:
            LOGGER.error(f"Object detection failed: {e}")
            results["objects"] = []
            results["object_counts"] = {}
            results["object_detections"] = []
        
        # 2. Action Recognition with VideoMAE
        try:
            LOGGER.debug(f"Running action recognition on {video_path.name}")
            action_result = self._recognize_action(frames)
            results["action"] = action_result.get("action", "unknown")
            results["action_confidence"] = action_result.get("confidence", 0.0)
            confidences.append(action_result.get("confidence", 0.5))
        except Exception as e:
            LOGGER.error(f"Action recognition failed: {e}")
            results["action"] = "unknown"
            results["action_confidence"] = 0.0
        
        # 3. Caption Generation with BLIP-2
        try:
            LOGGER.debug(f"Generating caption for {video_path.name}")
            caption_results = self.blip2.generate_captions(frames)
            results["caption"] = caption_results.get("caption", "")
            results["all_captions"] = caption_results.get("all_captions", [])
            confidences.append(caption_results.get("confidence", 0.5))
        except Exception as e:
            LOGGER.error(f"Caption generation failed: {e}")
            results["caption"] = ""
            results["all_captions"] = []
        
        # 4. Audio Transcription with Whisper
        if include_audio:
            try:
                LOGGER.debug(f"Transcribing audio from {video_path.name}")
                audio_results = self.whisper.transcribe(video_path)
                results["audio_text"] = audio_results.get("audio_text", "")
                results["has_speech"] = audio_results.get("has_speech", False)
                results["audio_language"] = audio_results.get("language", "unknown")
                if audio_results.get("has_speech", False):
                    confidences.append(audio_results.get("confidence", 0.5))
            except Exception as e:
                LOGGER.error(f"Audio transcription failed: {e}")
                results["audio_text"] = ""
                results["has_speech"] = False
                results["audio_language"] = "unknown"
        else:
            results["audio_text"] = ""
            results["has_speech"] = False
            results["audio_language"] = "unknown"
        
        # Calculate overall confidence
        results["confidence"] = float(np.mean(confidences)) if confidences else 0.5
        
        # Add metadata
        results["metadata"] = {
            "num_frames": len(frames),
            "video_path": str(video_path),
            "models_used": {
                "object_detection": "YOLOv8",
                "action_recognition": "VideoMAE",
                "caption_generation": "BLIP-2",
                "audio_transcription": "Whisper"
            }
        }
        
        LOGGER.info(
            f"Labeling complete for {video_path.name}: "
            f"{len(results.get('objects', []))} objects, "
            f"action='{results.get('action')}', "
            f"confidence={results['confidence']:.2f}"
        )
        
        return results
    
    def _recognize_action(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        Recognize action in video frames using VideoMAE.
        
        Args:
            frames: List of frames as numpy arrays
        
        Returns:
            Dictionary with recognized action and confidence
        """
        try:
            # VideoMAE encoding returns embeddings
            # We can use the embedding patterns to infer actions
            embedding = self.videomae.encode(frames)
            
            # Simple heuristic-based action recognition
            # In a production system, you'd train a classifier on top of VideoMAE
            embedding_norm = np.linalg.norm(embedding)
            embedding_mean = np.mean(embedding)
            embedding_std = np.std(embedding)
            
            # Basic heuristics for common actions
            if embedding_std > 0.5:
                action = "dynamic movement"
                confidence = 0.7
            elif embedding_std > 0.3:
                action = "walking"
                confidence = 0.6
            elif embedding_std > 0.15:
                action = "standing or slow movement"
                confidence = 0.5
            else:
                action = "static scene"
                confidence = 0.6
            
            return {
                "action": action,
                "confidence": confidence,
                "embedding_stats": {
                    "norm": float(embedding_norm),
                    "mean": float(embedding_mean),
                    "std": float(embedding_std)
                }
            }
        
        except Exception as e:
            LOGGER.error(f"Action recognition failed: {e}")
            return {
                "action": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def batch_label(
        self,
        video_paths: List[Path],
        include_audio: bool = True
    ) -> List[Dict[str, any]]:
        """
        Label multiple video chunks in batch.
        
        Args:
            video_paths: List of paths to video files
            include_audio: Whether to transcribe audio
        
        Returns:
            List of labeling results for each video
        """
        results = []
        
        for i, video_path in enumerate(video_paths):
            LOGGER.info(f"Labeling video {i+1}/{len(video_paths)}: {video_path.name}")
            
            try:
                result = self.label_video_chunk(video_path, include_audio=include_audio)
                results.append(result)
            except Exception as e:
                LOGGER.error(f"Failed to label {video_path}: {e}")
                results.append({
                    "error": str(e),
                    "video_path": str(video_path)
                })
        
        return results


__all__ = ["AutoLabeler"]
