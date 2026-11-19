"""Image-encoder interface and the tiny random stand-in used by CPU tests."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class ImageEncoder(nn.Module, ABC):
    """Maps pixels to a grid of patch tokens.

    Contract: ``forward(pixel_values)`` takes ``[B, 3, H, W]`` with H and W multiples of
    ``patch_size`` and returns ``[B, (H / patch_size) * (W / patch_size), output_dim]``.
    Prefix tokens such as CLS and registers are never returned.
    """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Channel size of each returned patch token."""

    @property
    @abstractmethod
    def patch_size(self) -> int:
        """Side of one square patch in pixels."""

    def num_patches(self, height: int, width: int) -> int:
        self.check_image_size(height, width)
        return (height // self.patch_size) * (width // self.patch_size)

    def check_image_size(self, height: int, width: int) -> None:
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"image sides must be multiples of {self.patch_size}, got {height}x{width}"
            )

    @abstractmethod
    def forward(self, pixel_values: Tensor) -> Tensor:
        """Return patch tokens ``[B, N, output_dim]``."""


class RandomImageEncoder(ImageEncoder):
    """Small randomly initialised encoder with the same interface as the DINOv3 wrapper.

    It exists only so that CPU tests can run the full pipeline without real weights.
    Its features mean nothing.
    """

    def __init__(self, output_dim: int = 64, patch_size: int = 16, seed: int = 0) -> None:
        super().__init__()
        self._output_dim = output_dim
        self._patch_size = patch_size
        generator = torch.Generator().manual_seed(seed)
        self.proj = nn.Conv2d(3, output_dim, kernel_size=patch_size, stride=patch_size)
        with torch.no_grad():
            for parameter in self.proj.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    def forward(self, pixel_values: Tensor) -> Tensor:
        self.check_image_size(pixel_values.shape[-2], pixel_values.shape[-1])
        return self.proj(pixel_values).flatten(2).transpose(1, 2)
