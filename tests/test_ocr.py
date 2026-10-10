# tests/test_ocr.py

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).parent.parent))

from ocr import ocr_engine
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


# --- worker count ----------------------------------------------------------------------

@pytest.mark.parametrize(("cpus", "expected"), [(1, 1), (2, 2), (4, 3), (8, 6), (16, 12), (64, 16)])
def test_default_workers_scale_with_the_machine(monkeypatch, cpus, expected):
    monkeypatch.delenv("OCR_WORKERS", raising=False)
    monkeypatch.setattr(ocr_engine, "load_dotenv", lambda: None)
    monkeypatch.setattr(ocr_engine, "_usable_cpus", lambda: cpus)

    assert ocr_engine.default_ocr_workers() == expected


def test_ocr_workers_env_override(monkeypatch):
    monkeypatch.setattr(ocr_engine, "load_dotenv", lambda: None)
    monkeypatch.setattr(ocr_engine, "_usable_cpus", lambda: 16)

    monkeypatch.setenv("OCR_WORKERS", "3")
    assert ocr_engine.default_ocr_workers() == 3
    monkeypatch.setenv("OCR_WORKERS", "lots")
    assert ocr_engine.default_ocr_workers() == 12  # invalid value: falls back to the rule
    monkeypatch.setenv("OCR_WORKERS", "0")
    assert ocr_engine.default_ocr_workers() == 12


# --- re-reading crops in parallel ---------------------------------------------------------

def _lines_image(lines: list[str]) -> Image.Image:
    img = Image.new("RGB", (900, 70 * len(lines) + 20), "white")
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((20, 20 + 70 * i), text, fill="black", font=ImageFont.load_default(size=36))
    return img


@requires_tesseract
def test_read_crops_keeps_order_and_maps_to_original_pixels():
    lines = ["Passport TK4820913", "SSN 900-42-7781", "Staff EMP-40944"]
    img = _lines_image(lines)
    boxes = [(10, 10 + 70 * i, 880, 75 + 70 * i) for i in range(len(lines))]

    results = OcrEngine()._read_crops(img, boxes, 3.0, psm=7)

    # Order is what matters here (exact digits depend on OCR of the test font).
    assert [results[i][0].text for i in range(len(lines))] == ["Passport", "SSN", "Staff"]
    for i, box in enumerate(boxes):
        for w in results[i]:
            assert box[0] <= w.left and w.left + w.width <= box[2]
            assert box[1] <= w.top and w.top + w.height <= box[3]


def test_failed_crop_is_left_out(monkeypatch):
    engine = OcrEngine()
    calls = []

    def fake_run(img, scale, psm=None):
        calls.append(img.size)
        if len(calls) == 2:
            raise RuntimeError("Tesseract process timeout")
        return []

    monkeypatch.setattr(engine, "_run", fake_run)
    results = engine._read_crops(Image.new("RGB", (100, 100), "white"),
                                 [(0, 0, 50, 20), (0, 30, 50, 50), (0, 60, 50, 80)], 2.0, psm=7)

    assert len(calls) == 3
    assert len(results) == 2  # the failed crop is missing, so its caller keeps the old words
