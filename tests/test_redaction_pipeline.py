
import pymupdf

from models.schemas import (
    BoundingBox,
    DetectedEntityDetail,
    ExtractionInput,
    PIIOutput,
)
from redaction.pipeline import RedactionPipeline


def create_test_data(tmp_path):
    input_pdf = tmp_path / "original.pdf"

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Employee: John Smith")
    doc.save(str(input_pdf))
    doc.close()

    raw_text = "Employee: John Smith"

    doc = pymupdf.open(str(input_pdf))
    page = doc[0]
    words = page.get_text("words")
    doc.close()

    boxes = []

    for word in words:
        text = word[4]

        if text in ("John", "Smith"):
            start = raw_text.index(text)

            boxes.append(
                BoundingBox(
                    page=1,
                    text=text,
                    x=word[0],
                    y=word[1],
                    w=word[2] - word[0],
                    h=word[3] - word[1],
                    char_start=start,
                    char_end=start + len(text),
                )
            )

    chunk = ExtractionInput(
        file_id="test",
        file_name="original.pdf",
        page_number=1,
        raw_text=raw_text,
        bounding_boxes=boxes,
    )

    finding = DetectedEntityDetail(
        token="[PERSON_001]",
        entity_type="PERSON",
        original_value="John Smith",
        start=10,
        end=20,
        score=0.99,
        page_number=1,
    )

    original_result = PIIOutput(
        file_id="test",
        file_name="original.pdf",
        page_number=1,
        status="SUCCESS",
        sanitized_text=raw_text,
        detected_entities=[finding],
        verification_passed=False,
    )

    clean_result = PIIOutput(
        file_id="test",
        file_name="sanitized.pdf",
        page_number=1,
        status="SUCCESS",
        sanitized_text="Employee: [REDACTED] [REDACTED]",
        detected_entities=[],
        verification_passed=True,
    )

    return input_pdf, chunk, original_result, clean_result


def test_pipeline_passes_when_rescan_finds_no_pii(tmp_path):
    input_pdf, chunk, original, clean = create_test_data(tmp_path)
    output_pdf = tmp_path / "sanitized.pdf"

    extraction_calls = []

    def fake_extractor(path):
        extraction_calls.append(str(path))
        clean_chunk = chunk.model_copy(
            update={
                "file_name": "sanitized.pdf",
                "raw_text": "Employee: [REDACTED] [REDACTED]",
                "bounding_boxes": [],
            }
        )
        return [clean_chunk]

    class FakePIIPipeline:
        def process_document_chunks(self, chunks):
            assert len(chunks) == 1
            return [clean]

    pipeline = RedactionPipeline(
        extractor=fake_extractor,
        pii_pipeline=FakePIIPipeline(),
    )

    result = pipeline.process(
        input_pdf=str(input_pdf),
        output_pdf=str(output_pdf),
        chunks=[chunk],
        pii_results=[original],
    )

    assert extraction_calls == [str(output_pdf)]
    assert result["status"] == "VERIFIED_NO_PII_FOUND"
    assert result["verification_passed"] is True
    assert output_pdf.exists()

    doc = pymupdf.open(str(output_pdf))
    output_text = "".join(page.get_text() for page in doc)
    doc.close()

    assert "John" not in output_text
    assert "Smith" not in output_text


def test_pipeline_blocks_when_rescan_finds_pii(tmp_path):
    input_pdf, chunk, original, _clean = create_test_data(tmp_path)
    output_pdf = tmp_path / "sanitized.pdf"

    def fake_extractor(path):
        return [
            chunk.model_copy(
                update={
                    "file_name": "sanitized.pdf",
                    "raw_text": "Employee: [REDACTED] [REDACTED]",
                    "bounding_boxes": [],
                }
            )
        ]

    class FakePIIPipeline:
        def process_document_chunks(self, chunks):
            return [original]

    pipeline = RedactionPipeline(
        extractor=fake_extractor,
        pii_pipeline=FakePIIPipeline(),
    )

    result = pipeline.process(
        input_pdf=str(input_pdf),
        output_pdf=str(output_pdf),
        chunks=[chunk],
        pii_results=[original],
    )

    assert result["status"] == "BLOCKED"
    assert result["verification_passed"] is False
    assert result["output_pdf"] is None
    assert not output_pdf.exists()
