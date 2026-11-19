import pytest
import torch

from mlis.data.reader import iter_samples
from mlis.models.image_encoder import RandomImageEncoder
from mlis.models.preprocess import GENERAL_SIZE, PERSON_SIZE, ImagePreprocessor


def test_random_encoder_returns_patch_tokens():
    encoder = RandomImageEncoder(output_dim=32, patch_size=16)
    out = encoder(torch.randn(2, 3, 64, 32))
    assert out.shape == (2, 4 * 2, 32)
    assert encoder.num_patches(64, 32) == 8


def test_random_encoder_is_frozen_and_deterministic():
    a, b = RandomImageEncoder(seed=3), RandomImageEncoder(seed=3)
    assert not any(p.requires_grad for p in a.parameters())
    x = torch.randn(1, 3, 32, 32)
    assert torch.equal(a(x), b(x))
    assert not torch.equal(a(x), RandomImageEncoder(seed=4)(x))


def test_rejects_sides_that_are_not_patch_multiples():
    with pytest.raises(ValueError, match="multiples of 16"):
        RandomImageEncoder()(torch.randn(1, 3, 30, 32))


def test_preprocessor_keeps_person_aspect_ratio_contract(shard_manifest):
    images = [s.image for s in iter_samples(shard_manifest)][:3]
    assert ImagePreprocessor(PERSON_SIZE)(images).shape == (3, 3, 256, 128)
    assert ImagePreprocessor(GENERAL_SIZE)(images).shape == (3, 3, 224, 224)


def test_preprocessor_normalises_with_imagenet_statistics():
    from PIL import Image

    white = Image.new("RGB", (16, 16), (255, 255, 255))
    out = ImagePreprocessor((16, 16))([white])
    expected = (1.0 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    assert torch.allclose(out[0, :, 0, 0], expected, atol=1e-5)


def test_preprocessor_rejects_non_patch_multiple_size():
    with pytest.raises(ValueError):
        ImagePreprocessor((250, 128))
