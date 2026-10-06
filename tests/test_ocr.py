# tests/test_ocr.py

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))

from ocr.ocr_engine import OcrEngine, mean_confidence

requires_tesseract = pytest.mark.skipif(
    not OcrEngine().is_available(), reason="Tesseract binary not installed"
)


def _text_image(text: str, fg="black", bg="white", size=(900, 120)) -> Image.Image:
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), text, fill=fg, font=ImageFont.load_default(size=40))
    return img


def test_missing_binary_is_unavailable():
    assert OcrEngine(tesseract_cmd="does-not-exist.exe").is_available() is False


def test_preprocess_inverts_dark_images():
    processed, _ = OcrEngine().preprocess(_text_image("x", fg="white", bg="black"))

    # After inversion the background (most pixels) is light.
    assert processed.getpixel((0, 0)) > 200


def test_preprocess_upscales_small_images():
    img = _text_image("x")
    processed, scale = OcrEngine().preprocess(img)

    assert scale == 2
    assert processed.size == (img.width * 2, img.height * 2)


def test_mean_confidence_empty():
    assert mean_confidence([]) == 0.0


@requires_tesseract
def test_reads_words_with_original_pixel_boxes():
    img = _text_image("Passport TK4820913")
    words = OcrEngine().ocr_image(img)

    assert [w.text for w in words] == ["Passport", "TK4820913"]
    for w in words:
        assert 0 <= w.left and w.left + w.width <= img.width
        assert 0 <= w.top and w.top + w.height <= img.height
        assert w.conf >= 0


@requires_tesseract
def test_reads_light_text_on_dark_background():
    words = OcrEngine().ocr_image(_text_image("SSN 900-42-7781", fg="white", bg="black"))

    assert "900-42-7781" in [w.text for w in words]
