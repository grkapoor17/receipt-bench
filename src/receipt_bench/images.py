"""Image handling for vision models."""
import io
from typing import Tuple

# Claude rejects images over 5 MB once base64-encoded (x4/3), and downscales
# anything past 2576px on the long edge anyway
MAX_IMAGE_BYTES = 3_700_000
MAX_LONG_EDGE = 2576


def media_type(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    raise ValueError("unsupported image format (expected JPEG, PNG, WebP or GIF)")


def prepare_image(data: bytes) -> Tuple[bytes, str]:
    """Return (bytes, media_type), re-encoding as JPEG only if the image is too big to send."""
    if len(data) <= MAX_IMAGE_BYTES:
        return data, media_type(data)

    from PIL import Image  # optional dependency: pip install receipt-bench[claude]

    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((MAX_LONG_EDGE, MAX_LONG_EDGE))
    for quality in (85, 70, 55):
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        if buf.tell() <= MAX_IMAGE_BYTES:
            break
    return buf.getvalue(), "image/jpeg"
