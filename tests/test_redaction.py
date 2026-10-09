import pymupdf

from models.schemas import (
    BoundingBox,
    DetectedEntityDetail,
    ExtractionInput,
    PIIOutput,
)
from redaction.pdf_redactor import PDFRedactor


def test_pdf_redaction(tmp_path):

    # 1. Create a dummy PDF
    input_pdf = tmp_path / "original.pdf"
    output_pdf = tmp_path / "sanitized.pdf"

    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)

    page.insert_text(
        (100, 100),
        "Employee: John Smith",
        fontsize=20,
    )

    doc.save(input_pdf)
    doc.close()

    # 2. Get the actual position of "John Smith"
    doc = pymupdf.open(input_pdf)
    page = doc[0]

    words = page.get_text("words")

    target = [word for word in words if word[4] in ["John", "Smith"]]

    assert len(target) == 2

    john = target[0]
    smith = target[1]

    # Character positions in:
    # "Employee: John Smith"
    #
    # Employee: = 10 characters
    # John       = positions 10-14
    # Smith      = positions 15-20

    boxes = [
        BoundingBox(
            page=1,
            text="John",
            x=john[0],
            y=john[1],
            w=john[2] - john[0],
            h=john[3] - john[1],
            char_start=10,
            char_end=14,
            source="native",
        ),
        BoundingBox(
            page=1,
            text="Smith",
            x=smith[0],
            y=smith[1],
            w=smith[2] - smith[0],
            h=smith[3] - smith[1],
            char_start=15,
            char_end=20,
            source="native",
        ),
    ]

    doc.close()

    # 3. Create extraction chunk
    chunk = ExtractionInput(
        file_id="test",
        file_name="original.pdf",
        page_number=1,
        raw_text="Employee: John Smith",
        bounding_boxes=boxes,
    )

    # 4. Create PII finding covering both words
    finding = DetectedEntityDetail(
        token="[PERSON_001]",
        entity_type="PERSON",
        original_value="John Smith",
        start=10,
        end=20,
        score=0.99,
        page_number=1,
    )

    result = PIIOutput(
        file_id="test",
        file_name="original.pdf",
        page_number=1,
        status="SUCCESS",
        sanitized_text="Employee: [PERSON_001]",
        detected_entities=[finding],
        verification_passed=True,
    )

    # 5. Run YOUR redactor
    count = PDFRedactor().redact(
        str(input_pdf),
        str(output_pdf),
        [chunk],
        [result],
    )

    # 6. Make sure two word boxes were redacted
    assert count == 2

    # 7. Verify the output PDF exists
    assert output_pdf.exists()

    # 8. Check that the original text is no longer present
    sanitized = pymupdf.open(output_pdf)
    text = sanitized[0].get_text()

    assert "John" not in text
    assert "Smith" not in text

    sanitized.close()
