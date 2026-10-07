# tests/test_extraction.py

import sys
from pathlib import Path

import pymupdf
import pytest

sys.path.append(str(Path(__file__).parent.parent))

from extraction.pdf_extractor import extract_pdf, render_overlays, save_chunks
from ocr.ocr_engine import OcrEngine

NATIVE_LINE = "Contact Jane Doe EMP-12345 jane@example.com"
IMAGE_LINES = ["Escalate to Rafael Duarte", "Email r.duarte@example.com", "Staff ID EMP-40944"]

requires_tesseract = pytest.mark.skipif(
    not OcrEngine().is_available(), reason="Tesseract binary not installed"
)


def _text_image_png(lines: list[str]) -> bytes:
    """Render lines of text to a PNG, so the text exists only as pixels."""
    src = pymupdf.open()
    page = src.new_page(width=420, height=40 + 30 * len(lines))
    for i, line in enumerate(lines):
        page.insert_text((20, 35 + 30 * i), line, fontsize=18)
    png = page.get_pixmap(dpi=200).tobytes("png")
    src.close()
    return png


@pytest.fixture
def sample_pdf(tmp_path) -> Path:
    doc = pymupdf.open()

    # Page 1: native text only
    page = doc.new_page()
    page.insert_text((72, 100), NATIVE_LINE, fontsize=12)

    # Page 2: text that exists only inside an image, plus a tiny icon
    page = doc.new_page()
    page.insert_image(pymupdf.Rect(72, 100, 492, 220), stream=_text_image_png(IMAGE_LINES))
    page.insert_image(pymupdf.Rect(500, 20, 520, 40), stream=_text_image_png(["x"]))

    # Page 3: blank
    doc.new_page()

    path = tmp_path / "sample.pdf"
    doc.save(path)
    doc.close()
    return path


def test_one_chunk_per_page_in_order(sample_pdf):
    chunks = extract_pdf(sample_pdf)

    assert [c.page_number for c in chunks] == [1, 2, 3]


def test_metadata_identical_across_pages(sample_pdf):
    chunks = extract_pdf(sample_pdf, source_url="storage/sample.pdf", user_id="u1")

    assert {c.file_id for c in chunks} == {chunks[0].file_id}
    assert {c.file_name for c in chunks} == {"sample.pdf"}
    assert {c.source_url for c in chunks} == {"storage/sample.pdf"}
    assert chunks[0].file_id.startswith("doc_")


def test_file_id_is_stable(sample_pdf):
    assert extract_pdf(sample_pdf)[0].file_id == extract_pdf(sample_pdf)[0].file_id


def test_native_page(sample_pdf):
    page1 = extract_pdf(sample_pdf)[0]

    assert page1.raw_text == NATIVE_LINE
    assert page1.extraction_method == "native"
    assert page1.ocr_used is False
    assert all(b.source == "native" and b.page == 1 for b in page1.bounding_boxes)


def test_char_offsets_match_raw_text(sample_pdf):
    for chunk in extract_pdf(sample_pdf):
        for box in chunk.bounding_boxes:
            assert chunk.raw_text[box.char_start:box.char_end] == box.text


def test_blank_page(sample_pdf):
    page3 = extract_pdf(sample_pdf)[2]

    assert page3.raw_text == ""
    assert page3.bounding_boxes == []


def test_tiny_image_skipped(sample_pdf):
    page2 = extract_pdf(sample_pdf)[1]

    assert page2.images_skipped_count >= 1


def test_without_tesseract_images_are_reported_not_hidden(sample_pdf):
    unavailable = OcrEngine(tesseract_cmd="does-not-exist.exe")
    page2 = extract_pdf(sample_pdf, ocr_engine=unavailable)[1]

    assert page2.ocr_used is False
    assert page2.images_skipped_count == 2


@requires_tesseract
def test_image_text_is_ocrd(sample_pdf):
    page2 = extract_pdf(sample_pdf)[1]

    assert page2.extraction_method == "ocr"
    assert page2.ocr_used is True
    assert page2.images_ocr_count == 1
    assert "r.duarte@example.com" in page2.raw_text
    assert "EMP-40944" in page2.raw_text
    assert page2.ocr_mean_confidence is not None


@requires_tesseract
def test_ocr_boxes_fall_inside_image(sample_pdf):
    page2 = extract_pdf(sample_pdf)[1]
    image_rect = pymupdf.Rect(72, 100, 492, 220)

    ocr_boxes = [b for b in page2.bounding_boxes if b.source == "ocr"]
    assert ocr_boxes
    for b in ocr_boxes:
        assert image_rect.contains(pymupdf.Rect(b.x, b.y, b.x + b.w, b.y + b.h))
        assert b.confidence is not None


@requires_tesseract
def test_output_feeds_pii_pipeline(sample_pdf):
    spacy = pytest.importorskip("spacy")
    if not spacy.util.is_package("en_core_web_lg"):
        pytest.skip("spaCy model en_core_web_lg not installed")
    from pii.pipeline import PIIPipeline

    chunks = extract_pdf(sample_pdf)
    results = PIIPipeline().process_document_chunks(chunks)

    assert len(results) == 3
    assert "jane@example.com" not in results[0].sanitized_text
    assert "r.duarte@example.com" not in results[1].sanitized_text


def test_render_overlays_writes_one_png_per_page(sample_pdf, tmp_path):
    chunks = extract_pdf(sample_pdf)
    out = render_overlays(sample_pdf, chunks, tmp_path / "overlay")

    assert [p.name for p in out] == ["page_01.png", "page_02.png", "page_03.png"]
    assert all(p.stat().st_size > 0 for p in out)


def test_render_overlays_respects_page_filter(sample_pdf, tmp_path):
    chunks = extract_pdf(sample_pdf)
    out = render_overlays(sample_pdf, chunks, tmp_path / "overlay", pages={2})

    assert [p.name for p in out] == ["page_02.png"]


def test_save_chunks_round_trips(sample_pdf, tmp_path):
    import json

    from pydantic import TypeAdapter

    from models.schemas import ExtractionInput

    chunks = extract_pdf(sample_pdf)
    path = save_chunks(chunks, tmp_path / "chunks.json")
    loaded = TypeAdapter(list[ExtractionInput]).validate_python(
        json.loads(path.read_text(encoding="utf-8"))
    )

    assert loaded == chunks
