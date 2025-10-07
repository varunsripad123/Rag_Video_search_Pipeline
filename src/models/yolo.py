"""YOLO object detection wrapper for video frames."""
from __future__ import annotations

import logging
from typing import Dict, List, Iterable

import numpy as np
import torch

from .base import BaseEncoder

LOGGER = logging.getLogger(__name__)

# Defer imports to avoid dependency issues
YOLOModel = None


def _try_import_yolo():
    """Try to import ultralytics YOLO at runtime."""
    global YOLOModel
    if YOLOModel is not None:
        return True
    
    try:
        from ultralytics import YOLO as _YOLO
        YOLOModel = _YOLO
        return True
    except Exception as e:
        LOGGER.warning(f"Could not import ultralytics YOLO: {e}")
        return False


class YOLODetector(BaseEncoder):
    """Detects objects and scenes in video frames using YOLOv8."""
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.25, **kwargs):
        """
        Initialize YOLO detector.
        
        Args:
            model_name: YOLO model variant (yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.)
            confidence_threshold: Minimum confidence for detections
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._use_fallback = False
    
    def load(self) -> None:
        """Load the YOLO model."""
        if not _try_import_yolo():
            LOGGER.warning("YOLO not available, using fallback detector")
            self._use_fallback = True
            return
        
        try:
            self.model = YOLOModel(self.model_name)
            self.model.to(self.device)
            LOGGER.info(f"Loaded YOLO model: {self.model_name}")
            self._use_fallback = False
        except Exception as e:
            LOGGER.warning(f"YOLO model unavailable, using fallback: {e}")
            self.model = None
            self._use_fallback = True
    
    @torch.no_grad()
    def detect_objects(self, frames: Iterable[np.ndarray]) -> Dict[str, any]:
        """
        Detect objects in video frames.
        
        Args:
            frames: List of frames as numpy arrays (H, W, C)
        
        Returns:
            Dictionary with detected objects, counts, and confidence scores
        """
        frames_list = list(frames)
        
        # Check if model is loaded before proceeding
        if self._use_fallback or self.model is None:
            return self._fallback_detect(frames_list)
        
        # Strip alpha channel if present
        frames_list = [
            frame[:, :, :3] if frame.shape[2] == 4 else frame
            for frame in frames_list
        ]
        
        try:
            # Run detection on all frames
            all_detections = []
            object_counts = {}
            max_confidence = {}
            
            for frame in frames_list:
                # YOLO expects RGB format
                results = self.model(frame, conf=self.confidence_threshold, verbose=False)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            class_name = result.names[cls_id]
                            
                            all_detections.append({
                                "object": class_name,
                                "confidence": conf,
                                "bbox": box.xyxy[0].cpu().numpy().tolist()
                            })
                            
                            # Track counts and max confidence
                            object_counts[class_name] = object_counts.get(class_name, 0) + 1
                            max_confidence[class_name] = max(
                                max_confidence.get(class_name, 0), conf
                            )
            
            # Get top objects by frequency
            sorted_objects = sorted(
                object_counts.items(), 
                key=lambda x: (x[1], max_confidence.get(x[0], 0)), 
                reverse=True
            )
            
            top_objects = [obj for obj, _ in sorted_objects[:10]]  # Top 10 objects
            avg_confidence = (
                sum(max_confidence.values()) / len(max_confidence) 
                if max_confidence else 0.0
            )
            
            return {
                "objects": top_objects,
                "object_counts": object_counts,
                "detections": all_detections[:20],  # Keep top 20 detailed detections
                "total_detections": len(all_detections),
                "avg_confidence": float(avg_confidence),
                "unique_objects": len(object_counts)
            }
        
        except Exception as e:
            LOGGER.error(f"YOLO detection failed: {e}")
            return self._fallback_detect(frames_list)
    
    def _fallback_detect(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """Fallback detection using basic heuristics."""
        LOGGER.debug("Using fallback object detection")
        
        # Simple color-based heuristics
        avg_brightness = np.mean([frame.mean() for frame in frames])
        
        # Guess based on brightness and color distribution
        objects = []
        if avg_brightness > 150:
            objects.append("person")
        if avg_brightness < 100:
            objects.append("indoor scene")
        else:
            objects.append("outdoor scene")
        
        return {
            "objects": objects,
            "object_counts": {obj: 1 for obj in objects},
            "detections": [],
            "total_detections": 0,
            "avg_confidence": 0.5,
            "unique_objects": len(objects),
            "fallback": True
        }
    
    def encode(self, frames: Iterable[np.ndarray]) -> np.ndarray:
        """Encode frames (required by BaseEncoder interface)."""
        # For YOLO, we return a simple embedding based on detected objects
        detection_result = self.detect_objects(frames)
        
        # Create a simple embedding: one-hot encoding of top objects
        embedding = np.zeros(512, dtype=np.float32)
        
        # Use hash of object names to create features
        for i, obj in enumerate(detection_result.get("objects", [])[:10]):
            idx = hash(obj) % 512
            embedding[idx] = 1.0
        
        # Normalize
        norm = np.linalg.norm(embedding) + 1e-8
        return (embedding / norm).reshape(1, -1)


__all__ = ["YOLODetector"]
