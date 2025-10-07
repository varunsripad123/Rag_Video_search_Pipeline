# src/models/videomaefeatureextractor.py

import numpy as np
from typing import List, Optional, Union
from PIL import Image


class VideoMAEFeatureExtractor:
    """
    Minimal VideoMAE feature extractor compatible with your pipeline.
    Performs resizing, normalization, and formatting expected by VideoMAE.
    """

    def __init__(
        self,
        size: int = 224,
        do_resize: bool = True,
        do_center_crop: bool = True,
        do_normalize: bool = True,
        image_mean: Optional[Union[List[float], float]] = None,
        image_std: Optional[Union[List[float], float]] = None,
        rescale_factor: float = 1 / 255,
    ):
        self.size = size
        self.do_resize = do_resize
        self.do_center_crop = do_center_crop
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.485, 0.456, 0.406]
        self.image_std = image_std if image_std is not None else [0.229, 0.224, 0.225]
        self.rescale_factor = rescale_factor

    def _resize(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((self.size, self.size), Image.Resampling.BILINEAR)
        return np.array(pil_image)

    def _center_crop(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        top = (h - self.size) // 2
        left = (w - self.size) // 2
        cropped = image[top : top + self.size, left : left + self.size]
        return cropped

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) * self.rescale_factor
        mean = np.array(self.image_mean)
        std = np.array(self.image_std)
        # Ensure broadcasting if mean/std are scalars
        if mean.ndim == 0:
            mean = np.full(image.shape[2:], mean)
        if std.ndim == 0:
            std = np.full(image.shape[2:], std)
        image = (image - mean) / std
        return image

    def __call__(self, videos: List[List[np.ndarray]], return_tensors: str = "pt") -> dict:
        """
        videos: List of videos, each video is a list of HWC numpy frames.
        Output tensor shape: (batch_size, num_frames, channels, height, width)
        """

        import torch

        processed_videos = []
        for video in videos:
            processed_frames = []
            for frame in video:
                if self.do_resize:
                    frame = self._resize(frame)
                if self.do_center_crop:
                    frame = self._center_crop(frame)
                if self.do_normalize:
                    frame = self._normalize(frame)
                # Convert HWC => CHW
                frame = frame.transpose(2, 0, 1)
                processed_frames.append(frame)
            video_frames = np.stack(processed_frames, axis=0)  # (num_frames, C, H, W)
            processed_videos.append(video_frames)

        batch = np.stack(processed_videos, axis=0)  # (batch_size, num_frames, C, H, W)

        if return_tensors == "pt":
            batch = torch.from_numpy(batch)
            return {"pixel_values": batch}
        elif return_tensors == "np":
            return {"pixel_values": batch}
        else:
            raise ValueError(f"Unsupported return_tensors value {return_tensors}, expected 'pt' or 'np'")
