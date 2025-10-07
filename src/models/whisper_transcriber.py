"""Whisper audio transcription wrapper for video files."""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

# Defer imports to avoid dependency issues
WhisperModel = None
WhisperProcessor = None


def _try_import_whisper():
    """Try to import Whisper at runtime."""
    global WhisperModel, WhisperProcessor
    if WhisperModel is not None:
        return True
    
    try:
        from transformers import (
            WhisperForConditionalGeneration as _WhisperModel,
            WhisperProcessor as _WhisperProcessor
        )
        WhisperModel = _WhisperModel
        WhisperProcessor = _WhisperProcessor
        return True
    except Exception as e:
        LOGGER.warning(f"Could not import Whisper: {e}")
        return False


class WhisperTranscriber:
    """Transcribes audio from video files using Whisper."""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-tiny",
        device: str = "cuda",
        language: Optional[str] = None
    ):
        """
        Initialize Whisper transcriber.
        
        Args:
            model_name: Whisper model variant (tiny, base, small, medium, large)
            device: Device to run on (cuda or cpu)
            language: Target language code (None for auto-detect)
        """
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.language = language
        self.model = None
        self.processor = None
        self._use_fallback = False
    
    def load(self) -> None:
        """Load the Whisper model and processor."""
        if not _try_import_whisper():
            LOGGER.warning("Whisper not available, using fallback transcriber")
            self._use_fallback = True
            return
        
        try:
            LOGGER.info(f"Loading Whisper model: {self.model_name}")
            
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            
            LOGGER.info("Whisper model loaded successfully")
            self._use_fallback = False
        except Exception as e:
            LOGGER.warning(f"Whisper model unavailable, using fallback: {e}")
            self.model = None
            self.processor = None
            self._use_fallback = True
    
    def extract_audio(self, video_path: Path) -> Optional[Path]:
        """
        Extract audio from video file using ffmpeg.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Path to extracted audio file (WAV format) or None if failed
        """
        try:
            # Create temporary WAV file
            temp_audio = Path(tempfile.mktemp(suffix=".wav"))
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # 16kHz sample rate (Whisper expects this)
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                str(temp_audio)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and temp_audio.exists():
                return temp_audio
            else:
                LOGGER.warning(f"ffmpeg audio extraction failed: {result.stderr}")
                return None
        
        except Exception as e:
            LOGGER.error(f"Audio extraction failed: {e}")
            return None
    
    @torch.no_grad()
    def transcribe(self, video_path: Path) -> Dict[str, any]:
        """
        Transcribe audio from video file.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with transcription and metadata
        """
        # Check if models are loaded before proceeding
        if self._use_fallback or self.model is None or self.processor is None:
            return self._fallback_transcribe()
        
        # Extract audio from video
        audio_path = self.extract_audio(video_path)
        if audio_path is None:
            return {
                "audio_text": "",
                "language": "unknown",
                "confidence": 0.0,
                "has_speech": False,
                "error": "Audio extraction failed"
            }
        
        try:
            # Load audio using processor
            import librosa
            
            # Load audio file
            audio_array, sampling_rate = librosa.load(
                str(audio_path), 
                sr=16000,
                mono=True
            )
            
            # Process audio
            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt"
            )
            inputs = inputs.input_features.to(
                self.device,
                torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            # Generate transcription
            predicted_ids = self.model.generate(inputs)
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            # Clean up temp audio file
            try:
                audio_path.unlink()
            except:
                pass
            
            # Determine if there's actual speech
            has_speech = len(transcription) > 0 and not transcription.lower() in [
                "[blank_audio]", "thank you.", "thanks for watching!"
            ]
            
            return {
                "audio_text": transcription,
                "language": self.language or "en",
                "confidence": 0.8 if has_speech else 0.3,
                "has_speech": has_speech,
                "audio_duration": len(audio_array) / 16000.0
            }
        
        except Exception as e:
            LOGGER.error(f"Whisper transcription failed: {e}")
            
            # Clean up temp audio file
            try:
                if audio_path and audio_path.exists():
                    audio_path.unlink()
            except:
                pass
            
            return self._fallback_transcribe()
    
    def _fallback_transcribe(self) -> Dict[str, any]:
        """Fallback transcription when Whisper is unavailable."""
        LOGGER.debug("Using fallback transcription")
        
        return {
            "audio_text": "",
            "language": "unknown",
            "confidence": 0.0,
            "has_speech": False,
            "fallback": True
        }


__all__ = ["WhisperTranscriber"]
