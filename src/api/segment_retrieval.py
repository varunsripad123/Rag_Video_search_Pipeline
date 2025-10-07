"""Advanced video segment retrieval with flexible options."""

from pathlib import Path
from typing import Optional, Literal
import subprocess
import tempfile
from fastapi import HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import logging

LOGGER = logging.getLogger(__name__)


class SegmentRetriever:
    """Retrieve video segments with various options."""
    
    @staticmethod
    def extract_segment(
        video_path: Path,
        start_time: float,
        end_time: float,
        output_format: Literal["mp4", "gif", "thumbnail"] = "mp4",
        quality: Literal["high", "medium", "low"] = "medium"
    ) -> Path:
        """
        Extract a specific segment from a video.
        
        Args:
            video_path: Path to original video
            start_time: Start time in seconds
            end_time: End time in seconds
            output_format: Output format (mp4, gif, thumbnail)
            quality: Quality level
            
        Returns:
            Path to extracted segment
        """
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
        
        # Create temporary output file
        suffix = f".{output_format}"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        output_path = Path(temp_file.name)
        temp_file.close()
        
        try:
            if output_format == "mp4":
                return SegmentRetriever._extract_mp4(
                    video_path, start_time, end_time, output_path, quality
                )
            elif output_format == "gif":
                return SegmentRetriever._extract_gif(
                    video_path, start_time, end_time, output_path
                )
            elif output_format == "thumbnail":
                return SegmentRetriever._extract_thumbnail(
                    video_path, start_time, output_path
                )
        except Exception as e:
            if output_path.exists():
                output_path.unlink()
            raise HTTPException(status_code=500, detail=f"Segment extraction failed: {e}")
    
    @staticmethod
    def _extract_mp4(
        video_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
        quality: str
    ) -> Path:
        """Extract MP4 segment."""
        duration = end_time - start_time
        
        # Quality settings
        quality_settings = {
            "high": ["-crf", "18"],
            "medium": ["-crf", "23"],
            "low": ["-crf", "28"]
        }
        
        cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            *quality_settings[quality],
            "-c:a", "aac",
            "-movflags", "+faststart",
            "-y",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
        
        return output_path
    
    @staticmethod
    def _extract_gif(
        video_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path
    ) -> Path:
        """Extract GIF segment."""
        duration = end_time - start_time
        
        cmd = [
            "ffmpeg",
            "-ss", str(start_time),
            "-i", str(video_path),
            "-t", str(duration),
            "-vf", "fps=10,scale=480:-1:flags=lanczos",
            "-y",
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
        
        return output_path
    
    @staticmethod
    def _extract_thumbnail(
        video_path: Path,
        timestamp: float,
        output_path: Path
    ) -> Path:
        """Extract thumbnail at specific timestamp."""
        cmd = [
            "ffmpeg",
            "-ss", str(timestamp),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            "-y",
            str(output_path.with_suffix(".jpg"))
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg failed: {result.stderr}")
        
        return output_path.with_suffix(".jpg")
    
    @staticmethod
    def get_segment_context(
        video_path: Path,
        start_time: float,
        end_time: float,
        context_seconds: float = 5.0
    ) -> tuple[float, float]:
        """
        Get segment with additional context before/after.
        
        Args:
            video_path: Path to video
            start_time: Original start time
            end_time: Original end time
            context_seconds: Seconds to add before/after
            
        Returns:
            (new_start, new_end) with context
        """
        # Get video duration
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        
        # Add context
        new_start = max(0, start_time - context_seconds)
        new_end = min(duration, end_time + context_seconds)
        
        return new_start, new_end


__all__ = ["SegmentRetriever"]
