from models.schemas import (
    DetectedEntityDetail,
    PIIOutput,
)
from redaction.verifier import RedactionVerifier


def test_verification_passes_when_no_pii_remains():

    result = PIIOutput(
        file_id="test",
        file_name="sanitized.pdf",
        page_number=1,
        status="SUCCESS",
        sanitized_text="This document is safe.",
        detected_entities=[],
        verification_passed=True,
    )

    assert RedactionVerifier().verify([result], expected_pages=1) is True


def test_verification_fails_when_pii_remains():

    finding = DetectedEntityDetail(
        token="[EMAIL_ADDRESS_001]",
        entity_type="EMAIL_ADDRESS",
        original_value="john@example.com",
        start=10,
        end=26,
        score=0.99,
        page_number=1,
    )

    result = PIIOutput(
        file_id="test",
        file_name="sanitized.pdf",
        page_number=1,
        status="SUCCESS",
        sanitized_text="Email: john@example.com",
        detected_entities=[finding],
        verification_passed=True,
    )

    assert RedactionVerifier().verify([result], expected_pages=1) is False


def test_verification_fails_when_results_are_empty():
    assert RedactionVerifier().verify([], expected_pages=1) is False


def test_verification_fails_when_a_page_is_missing():
    result = PIIOutput(
        file_id="test",
        file_name="sanitized.pdf",
        page_number=1,
        status="SUCCESS",
        sanitized_text="Page one",
        detected_entities=[],
        verification_passed=True,
    )

    assert RedactionVerifier().verify([result], expected_pages=2) is False


def test_verification_fails_when_page_is_duplicated():
    page_one = PIIOutput(
        file_id="test",
        file_name="sanitized.pdf",
        page_number=1,
        status="SUCCESS",
        sanitized_text="Page one",
        detected_entities=[],
        verification_passed=True,
    )

    duplicate_page_one = PIIOutput(
        file_id="test",
        file_name="sanitized.pdf",
        page_number=1,
        status="SUCCESS",
        sanitized_text="Duplicate",
        detected_entities=[],
        verification_passed=True,
    )

    assert (
        RedactionVerifier().verify([page_one, duplicate_page_one], expected_pages=2)
        is False
    )
