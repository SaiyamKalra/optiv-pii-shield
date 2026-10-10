
import pymupdf

from models.schemas import ExtractionInput
from redaction.pipeline import RedactionPipeline


def test_pipeline_blocks_when_extraction_page_is_missing(tmp_path):
    input_pdf = tmp_path / "three_pages.pdf"

    doc = pymupdf.open()
    for number in range(1, 4):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page {number}")
    doc.save(str(input_pdf))
    doc.close()

    # Deliberately omit page 2.
    chunks = [
        ExtractionInput(
            file_id="test",
            file_name="three_pages.pdf",
            page_number=1,
            raw_text="Page 1",
        ),
        ExtractionInput(
            file_id="test",
            file_name="three_pages.pdf",
            page_number=3,
            raw_text="Page 3",
        ),
    ]

    output_pdf = tmp_path / "sanitized.pdf"

    result = RedactionPipeline().process(
        input_pdf=str(input_pdf),
        output_pdf=str(output_pdf),
        chunks=chunks,
        pii_results=[],
    )

    assert result["status"] == "BLOCKED"
    assert result["verification_passed"] is False
    assert result["output_pdf"] is None
    assert not output_pdf.exists()
    assert result["inspection_issues"][0]["expected_pages"] == [1, 2, 3]
    assert result["inspection_issues"][0]["received_pages"] == [1, 3]
