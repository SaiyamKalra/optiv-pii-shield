import pymupdf

from extraction.pdf_extractor import extract_pdf
from pii.pipeline import PIIPipeline
from models.schemas import DetectedEntityDetail, PIIOutput
from redaction.pdf_redactor import PDFRedactor
from redaction.verifier import RedactionVerifier


def test_real_redaction_and_pii_rescan(tmp_path):
    input_pdf = tmp_path / "original.pdf"
    output_pdf = tmp_path / "sanitized.pdf"

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Contact: alexander.quinn@example.com")
    doc.save(str(input_pdf))
    doc.close()

    chunks = extract_pdf(input_pdf)
    assert len(chunks) == 1
    chunk = chunks[0]

    email = "alexander.quinn@example.com"
    start = chunk.raw_text.index(email)
    end = start + len(email)

    finding = DetectedEntityDetail(
        token="[EMAIL_001]",
        entity_type="EMAIL_ADDRESS",
        original_value=email,
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

    redaction_count = PDFRedactor().redact(
        input_pdf=str(input_pdf),
        output_pdf=str(output_pdf),
        chunks=chunks,
        pii_results=[original_result],
    )

    assert redaction_count > 0
    assert output_pdf.exists()

    rescanned_chunks = extract_pdf(output_pdf)
    pii_pipeline = PIIPipeline()
    rescanned_results = pii_pipeline.process_document_chunks(rescanned_chunks)

    rescanned_text = "\n".join(chunk.raw_text for chunk in rescanned_chunks)
    assert email not in rescanned_text

    verifier = RedactionVerifier()
    passed = verifier.verify(
        rescanned_results,
        expected_pages=len(chunks),
    )

    assert passed is True
    assert all(not result.detected_entities for result in rescanned_results)
