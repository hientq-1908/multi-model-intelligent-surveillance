"""Frozen DINOv3 ViT-L/16 wrapper loaded through Hugging Face transformers."""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import AutoModel, PreTrainedModel

from mlis.models.image_encoder import ImageEncoder

DINOV3_VITL16 = "facebook/dinov3-vitl16-pretrain-lvd1689m"


class Dinov3Encoder(ImageEncoder):
    """Frozen DINOv3 returning patch tokens only.

    The Hugging Face output layout is ``[CLS, registers..., patches...]``, so the first
    ``1 + num_register_tokens`` tokens are dropped.
    The model stays in ``eval()`` mode for good, because DINOv3 jitters its RoPE
    coordinates in train mode, and takes no gradients.
    Use bf16 or fp32: DINOv3 does not support fp16.
    """

    def __init__(self, model: PreTrainedModel) -> None:
        super().__init__()
        self.model = model
        config = model.config
        self._output_dim = int(config.hidden_size)
        self._patch_size = int(config.patch_size)
        self._num_prefix_tokens = 1 + int(getattr(config, "num_register_tokens", 0))
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str = DINOV3_VITL16,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Dinov3Encoder:
        """Load from the Hub (the repository is gated: accept its terms with your user first)."""
        if dtype == torch.float16:
            raise ValueError("DINOv3 does not support fp16; use bfloat16 or float32")
        return cls(AutoModel.from_pretrained(name_or_path, torch_dtype=dtype))

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def patch_size(self) -> int:
        return self._patch_size

    @property
    def num_prefix_tokens(self) -> int:
        return self._num_prefix_tokens

    def train(self, mode: bool = True) -> Dinov3Encoder:
        """Ignore requests for train mode; the frozen encoder always stays in eval mode."""
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, pixel_values: Tensor) -> Tensor:
        self.check_image_size(pixel_values.shape[-2], pixel_values.shape[-1])
        param = next(self.model.parameters())
        hidden = self.model(pixel_values=pixel_values.to(param.dtype)).last_hidden_state
        return hidden[:, self._num_prefix_tokens :, :]
