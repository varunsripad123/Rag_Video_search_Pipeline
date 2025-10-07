"""Model wrappers for embedding extraction and auto-labeling."""

from .clip import CLIPEncoder
from .videomae import VideoMAEEncoder
from .videoswin import VideoSwinEncoder

# Auto-labeling models (imported on-demand to avoid dependencies)
try:
    from .yolo import YOLODetector
    from .blip2 import BLIP2Captioner
    from .whisper_transcriber import WhisperTranscriber
    __all__ = [
        "CLIPEncoder", 
        "VideoMAEEncoder", 
        "VideoSwinEncoder",
        "YOLODetector",
        "BLIP2Captioner",
        "WhisperTranscriber"
    ]
except ImportError:
    # Auto-labeling models not available
    __all__ = ["CLIPEncoder", "VideoMAEEncoder", "VideoSwinEncoder"]
