import pymupdf

from extraction.pdf_extractor import extract_pdf
from models.schemas import DetectedEntityDetail, PIIOutput
from pii.pipeline import PIIPipeline
from redaction.pdf_redactor import PDFRedactor
from redaction.verifier import RedactionVerifier


def test_multipage_pdf_redacts_pii_and_rescans_every_page(tmp_path):
    input_pdf = tmp_path / "multipage_original.pdf"
    output_pdf = tmp_path / "multipage_sanitized.pdf"

    doc = pymupdf.open()

    page = doc.new_page()
    page.insert_text((72, 72), "Contact: alice@example.com")

    doc.new_page()
    doc[1].insert_text((72, 72), "This page contains no contact details.")

    page = doc.new_page()
    page.insert_text((72, 72), "Contact: bob@example.com")

    doc.save(str(input_pdf))
    doc.close()

    chunks = extract_pdf(input_pdf)
    assert len(chunks) == 3

    pii_pipeline = PIIPipeline()
    original_results = pii_pipeline.process_document_chunks(chunks)

    expected_emails = {
        "alice@example.com",
        "bob@example.com",
    }

    found_emails = {
        finding.original_value.lower()
        for result in original_results
        for finding in result.detected_entities
        if finding.entity_type == "EMAIL_ADDRESS"
    }

    assert expected_emails.issubset(found_emails)

    redaction_count = PDFRedactor().redact(
        input_pdf=str(input_pdf),
        output_pdf=str(output_pdf),
        chunks=chunks,
        pii_results=original_results,
    )
    assert redaction_count > 0

    rescanned_chunks = extract_pdf(output_pdf)
    assert len(rescanned_chunks) == 3

    rescanned_text = "\n".join(chunk.raw_text for chunk in rescanned_chunks)
    assert "alice@example.com" not in rescanned_text
    assert "bob@example.com" not in rescanned_text

    rescanned_results = pii_pipeline.process_document_chunks(rescanned_chunks)

    verifier = RedactionVerifier()

    assert verifier.verify(
        rescanned_results,
        expected_pages=3,
    )

    assert not verifier.verify(
        rescanned_results[:-1],
        expected_pages=3,
    )
