import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from mlis.models.dinov3 import DINOV3_VITL16, Dinov3Encoder


def tiny_encoder(registers: int = 4) -> Dinov3Encoder:
    config = DINOv3ViTConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        patch_size=16,
        num_register_tokens=registers,
    )
    return Dinov3Encoder(DINOv3ViTModel(config))


def test_drops_cls_and_register_tokens():
    encoder = tiny_encoder(registers=4)
    out = encoder(torch.randn(2, 3, 32, 48))
    assert encoder.num_prefix_tokens == 5
    assert out.shape == (2, 2 * 3, 32)


def test_non_square_input_gives_row_major_patch_grid():
    encoder = tiny_encoder()
    assert encoder(torch.randn(1, 3, 64, 32)).shape[1] == encoder.num_patches(64, 32) == 8


def test_is_frozen_and_stays_in_eval_mode():
    encoder = tiny_encoder()
    assert not any(p.requires_grad for p in encoder.parameters())
    encoder.train()
    assert not encoder.model.training


def test_forward_builds_no_graph():
    out = tiny_encoder()(torch.randn(1, 3, 32, 32))
    assert not out.requires_grad


def test_casts_input_to_model_dtype():
    encoder = tiny_encoder()
    encoder.model.to(torch.bfloat16)
    assert encoder(torch.randn(1, 3, 32, 32)).dtype == torch.bfloat16


def test_rejects_fp16_and_bad_size():
    with pytest.raises(ValueError, match="fp16"):
        Dinov3Encoder.from_pretrained(DINOV3_VITL16, dtype=torch.float16)
    with pytest.raises(ValueError, match="multiples of 16"):
        tiny_encoder()(torch.randn(1, 3, 20, 32))


@pytest.mark.network
@pytest.mark.slow
def test_real_dinov3_vitl16_shapes():
    encoder = Dinov3Encoder.from_pretrained()
    out = encoder(torch.randn(1, 3, 256, 128))
    assert encoder.output_dim == 1024
    assert out.shape == (1, 16 * 8, 1024)
