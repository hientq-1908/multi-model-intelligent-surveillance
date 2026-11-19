"""Image decoding shared by every reader."""

from __future__ import annotations

import io

from PIL import Image

from mlis.data.schema import ImageRef


def decode_image(data: bytes, ref: ImageRef | None = None) -> Image.Image:
    """Decode bytes to an RGB image and apply the crop box of ``ref`` when it has one."""
    with Image.open(io.BytesIO(data)) as opened:
        image = opened.convert("RGB")
    if ref is not None and ref.box is not None:
        x0, y0, x1, y1 = ref.box
        width, height = image.size
        if x1 > width or y1 > height:
            raise ValueError(f"box {ref.box} lies outside the {width}x{height} image")
        image = image.crop((round(x0), round(y0), round(x1), round(y1)))
    return image
