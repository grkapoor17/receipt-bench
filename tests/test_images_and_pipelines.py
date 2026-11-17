import io
import json
import os
from types import SimpleNamespace

import pytest
from PIL import Image

from receipt_bench import images, pipelines

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def test_media_type_detection():
    assert images.media_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert images.media_type(PNG) == "image/png"
    assert images.media_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    with pytest.raises(ValueError):
        images.media_type(b"%PDF-1.7")


def test_small_images_pass_through_untouched():
    assert images.prepare_image(PNG) == (PNG, "image/png")


def test_large_images_are_shrunk_to_jpeg():
    # random noise doesn't compress, so this PNG is ~27 MB
    img = Image.frombytes("RGB", (3000, 3000), os.urandom(3000 * 3000 * 3))
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=0)
    data, media_type = images.prepare_image(buf.getvalue())
    assert media_type == "image/jpeg"
    assert len(data) <= images.MAX_IMAGE_BYTES
    assert max(Image.open(io.BytesIO(data)).size) <= images.MAX_LONG_EDGE


# ------------------------------------------------------------------ pipelines


def textract_client(total):
    def field(t, v):
        return {"Type": {"Text": t}, "ValueDetection": {"Text": v}}

    doc = {
        "SummaryFields": [field("SUBTOTAL", "20.00"), field("TAX", "1.60"), field("TOTAL", total)],
        "LineItemGroups": [{"LineItems": [{"LineItemExpenseFields": [field("ITEM", "Pizza"), field("PRICE", "20.00")]}]}],
    }
    return SimpleNamespace(analyze_expense=lambda Document: {"ExpenseDocuments": [doc]})


class ExplodingClaude:
    @property
    def messages(self):
        raise AssertionError("Claude should not be called when Textract adds up")


def claude_client(reading):
    reply = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=json.dumps(reading))],
                            usage=SimpleNamespace(model_dump=lambda: {}))
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: reply))


CLAUDE_READING = {
    "items": [{"description": "Pizza", "quantity": None, "unit_price": None, "total_amount": 20.0}],
    "total_net": 20.0, "total_tax": 1.6, "tax_breakdown": [], "discounts": [], "tip": 0.0, "total_amount": 21.6,
    "supplier_name": "", "supplier_address": "", "date": "", "time": "", "currency": "USD",
    "adds_up": True, "notes": "",
}


def test_textract_only():
    result = pipelines.textract_only(PNG, textract_client=textract_client("21.60"))
    assert result.adds_up and not result.used_fallback


def test_repair_is_skipped_when_textract_adds_up():
    result = pipelines.textract_with_claude_repair(PNG, textract_client("21.60"), ExplodingClaude())
    assert result.adds_up and not result.used_fallback


def test_repair_replaces_a_bad_textract_reading():
    # Textract misread the total as 27.60
    result = pipelines.textract_with_claude_repair(PNG, textract_client("27.60"), claude_client(CLAUDE_READING))
    assert result.used_fallback and result.adds_up
    assert result.receipt["total_amount"] == 21.6


def test_textract_reading_stands_when_repair_also_fails():
    bad = dict(CLAUDE_READING, total_amount=30.0)
    result = pipelines.textract_with_claude_repair(PNG, textract_client("27.60"), claude_client(bad))
    assert result.used_fallback and not result.adds_up
    assert result.receipt["total_amount"] == 27.6
