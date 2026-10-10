
import pymupdf

from extraction.pdf_extractor import extract_pdf
from models.schemas import DetectedEntityDetail, PIIOutput
from redaction.pipeline import RedactionPipeline


def test_pipeline_blocks_when_pii_has_no_bounding_box(tmp_path):
    input_pdf = tmp_path / "original.pdf"
    output_pdf = tmp_path / "sanitized.pdf"

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Confidential employee record")
    doc.save(str(input_pdf))
    doc.close()

    chunks = extract_pdf(input_pdf)
    chunk = chunks[0]

    finding = DetectedEntityDetail(
        token="[PERSON_001]",
        entity_type="PERSON",
        original_value="Alex Example",
        start=1000,
        end=1012,
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

    result = RedactionPipeline().process(
        input_pdf=str(input_pdf),
        output_pdf=str(output_pdf),
        chunks=chunks,
        pii_results=[original_result],
    )

    assert result["status"] == "BLOCKED"
    assert result["verification_passed"] is False
    assert result["unmapped_findings"]
    assert result["output_pdf"] is None
    assert not output_pdf.exists()
    assert result["redaction_count"] == 0
    assert result["unmapped_findings"]
