import pymupdf

from extraction.pdf_extractor import extract_pdf
from models.schemas import (
    BoundingBox,
    ExtractionInput,
    DetectedEntityDetail,
    PIIOutput,
)
from redaction.pdf_redactor import PDFRedactor


def test_redacted_pdf_has_no_original_pii_text(tmp_path):
    input_pdf = tmp_path / "original.pdf"
    output_pdf = tmp_path / "sanitized.pdf"

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Employee: John Smith")
    doc.save(str(input_pdf))
    doc.close()

    chunks = extract_pdf(input_pdf)
    chunk = chunks[0]

    start = chunk.raw_text.index("John")
    end = chunk.raw_text.index("Smith") + len("Smith")

    finding = DetectedEntityDetail(
        token="[PERSON_001]",
        entity_type="PERSON",
        original_value="John Smith",
        start=start,
        end=end,
        score=0.99,
        page_number=1,
    )

    original_result = PIIOutput(
        file_id=chunk.file_id,
        file_name=chunk.file_name,
        page_number=1,
        status="SUCCESS",
        sanitized_text=chunk.raw_text,
        detected_entities=[finding],
        verification_passed=False,
    )

    PDFRedactor().redact(
        input_pdf=str(input_pdf),
        output_pdf=str(output_pdf),
        chunks=chunks,
        pii_results=[original_result],
    )

    rescanned_chunks = extract_pdf(output_pdf)
    rescanned_text = "\n".join(c.raw_text for c in rescanned_chunks)

    assert len(rescanned_chunks) == len(chunks)
    assert "John" not in rescanned_text
    assert "Smith" not in rescanned_text
