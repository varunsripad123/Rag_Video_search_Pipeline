"""Model wrappers for embedding extraction."""

from .clip import CLIPEncoder
from .videomae import VideoMAEEncoder
from .videoswin import VideoSwinEncoder

__all__ = ["CLIPEncoder", "VideoMAEEncoder", "VideoSwinEncoder"]
