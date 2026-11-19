"""Pixel preprocessing that keeps the aspect ratio contract of the image encoders."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from PIL import Image
from torch import Tensor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GENERAL_SIZE = (224, 224)
PERSON_SIZE = (256, 128)


@dataclass(frozen=True)
class ImagePreprocessor:
    """Resize to a fixed ``(height, width)`` and normalise with the ImageNet statistics.

    The Hugging Face DINOv3 processor squashes every image to 224x224 by default.
    Choose ``size`` per source instead, for example ``PERSON_SIZE`` for tall person crops.
    Both sides must be multiples of the encoder patch size.
    """

    size: tuple[int, int] = GENERAL_SIZE
    patch_size: int = 16

    def __post_init__(self) -> None:
        height, width = self.size
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"size {self.size} must be multiples of patch size {self.patch_size}")

    def __call__(self, images: Sequence[Image.Image]) -> Tensor:
        height, width = self.size
        tensors = []
        for image in images:
            resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
            data = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8)
            tensors.append(data.view(height, width, 3).permute(2, 0, 1))
        batch = torch.stack(tensors).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        return (batch - mean) / std
