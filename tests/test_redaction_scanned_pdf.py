import pymupdf
from PIL import Image, ImageDraw, ImageFont
from pii.pipeline import PIIPipeline
from extraction.pdf_extractor import extract_pdf
from redaction.pdf_redactor import PDFRedactor
from redaction.verifier import RedactionVerifier


def test_scanned_pdf_redacts_email_from_image(tmp_path):
    input_pdf = tmp_path / "scanned_original.pdf"
    output_pdf = tmp_path / "scanned_sanitized.pdf"

    image = Image.new("RGB", (1200, 300), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (50, 100),
        "Contact: alexander.quinn@example.com",
        fill="black",
        font=ImageFont.truetype("arial.ttf", 42),
    )

    image_path = tmp_path / "scanned_page.png"
    image.save(image_path)

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=150)
    page.insert_image(page.rect, filename=str(image_path))
    doc.save(str(input_pdf))
    doc.close()

    chunks = extract_pdf(input_pdf)
    assert len(chunks) == 1
    chunk = chunks[0]

    email = "alexander.quinn@example.com"
    start = chunk.raw_text.find(email)

    assert start >= 0, f"OCR did not recover the email. Text: {chunk.raw_text!r}"
    end = start + len(email)

    assert any(
        box.source == "ocr" and box.char_start is not None and box.char_end is not None
        for box in chunk.bounding_boxes or []
    ), "Expected OCR-derived bounding boxes"

    pii_pipeline = PIIPipeline()
    original_results = pii_pipeline.process_document_chunks(chunks)

    findings = [
        finding
        for result in original_results
        for finding in result.detected_entities
        if finding.entity_type == "EMAIL_ADDRESS"
        and email.lower() in finding.original_value.lower()
    ]

    assert findings, (
        "The real PII pipeline did not detect the email in the scanned PDF. "
        f"OCR text: {chunk.raw_text!r}"
    )

    redaction_count = PDFRedactor().redact(
        input_pdf=str(input_pdf),
        output_pdf=str(output_pdf),
        chunks=chunks,
        pii_results=original_results,
    )

    assert redaction_count > 0
    assert output_pdf.exists()

    rescanned_chunks = extract_pdf(output_pdf)
    assert len(rescanned_chunks) == len(chunks)

    rescanned_text = "\n".join(c.raw_text for c in rescanned_chunks)
    assert email not in rescanned_text

    rescanned_results = pii_pipeline.process_document_chunks(rescanned_chunks)

    assert RedactionVerifier().verify(
        rescanned_results,
        expected_pages=len(chunks),
    )
