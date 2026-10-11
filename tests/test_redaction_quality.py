from pathlib import Path

import pymupdf

from models.schemas import ExtractionInput, PIIOutput
from redaction.pipeline import RedactionPipeline


def make_clean_result():
    return PIIOutput(
        file_id="test",
        file_name="clean.pdf",
        page_number=1,
        status="SUCCESS",
        sanitized_text="Ordinary document text",
        detected_entities=[],
        verification_passed=True,
    )


def make_chunk(tmp_path, **overrides):
    pdf_path = tmp_path / "input.pdf"

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Ordinary document text")
    doc.save(str(pdf_path))
    doc.close()

    values = {
        "file_id": "test",
        "file_name": "input.pdf",
        "page_number": 1,
        "raw_text": "Ordinary document text",
        "extraction_method": "native",
    }
    values.update(overrides)

    return pdf_path, ExtractionInput(**values)


def run_pipeline(tmp_path, chunk):
    output_path = tmp_path / "output.pdf"

    def fake_extractor(path):
        return [
            chunk.model_copy(
                update={
                    "file_name": "output.pdf",
                    "raw_text": "Ordinary document text",
                    "bounding_boxes": [],
                    "images_skipped_count": 0,
                }
            )
        ]

    class FakePIIPipeline:
        def process_document_chunks(self, chunks):
            return [make_clean_result()]

    result = RedactionPipeline(
        extractor=fake_extractor,
        pii_pipeline=FakePIIPipeline(),
    ).process(
        input_pdf=str(tmp_path / "input.pdf"),
        output_pdf=str(output_path),
        chunks=[chunk],
        pii_results=[],
    )

    return result


def test_native_text_document_can_pass(tmp_path):
    _, chunk = make_chunk(tmp_path)

    result = run_pipeline(tmp_path, chunk)

    assert result["status"] == "VERIFIED_NO_PII_FOUND"
    assert result["inspection_issues"] == []
    assert result["inspection_warnings"] == []


def test_ocr_without_confidence_is_blocked(tmp_path):
    _, chunk = make_chunk(
        tmp_path,
        extraction_method="ocr",
        ocr_used=True,
        ocr_mean_confidence=None,
    )

    result = run_pipeline(tmp_path, chunk)

    assert result["status"] == "BLOCKED"
    assert result["output_pdf"] is None
    assert not (tmp_path / "output.pdf").exists()
    assert result["redaction_count"] == 0
    assert result["redaction_count"] == 0
    assert any(
        "confidence is unavailable" in issue["reason"]
        for issue in result["inspection_issues"]
    )


def test_low_confidence_ocr_is_blocked(tmp_path):
    _, chunk = make_chunk(
        tmp_path,
        extraction_method="ocr",
        ocr_used=True,
        ocr_mean_confidence=85.0,
        low_confidence_word_count=2,
    )

    result = run_pipeline(tmp_path, chunk)

    assert result["status"] == "BLOCKED"
    assert result["output_pdf"] is None
    assert not (tmp_path / "output.pdf").exists()
    assert result["redaction_count"] == 0
    assert any(
        "low-confidence" in issue["reason"] for issue in result["inspection_issues"]
    )


def test_skipped_images_require_review(tmp_path):
    _, chunk = make_chunk(
        tmp_path,
        images_skipped_count=1,
    )

    result = run_pipeline(tmp_path, chunk)

    assert result["status"] == "REVIEW_REQUIRED"
    assert result["inspection_warnings"]
