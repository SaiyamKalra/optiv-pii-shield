from models.schemas import (
    DetectedEntityDetail,
    ExtractionInput,
    PIIOutput,
    UnreadRegion,
)
from ui import components
from ui.evidence import attention, build_trail, summarize


def _res(status, **extra):
    chunk = ExtractionInput(file_id="doc_abc", file_name="a.pdf", raw_text="Contact Helena Vorstermann")
    finding = DetectedEntityDetail(token="[PERSON_001]", entity_type="PERSON",
                                   original_value="Helena Vorstermann", start=8, end=26, score=0.9)
    output = PIIOutput(file_id="doc_abc", file_name="a.pdf", status="SUCCESS", sanitized_text="",
                       detected_entities=[finding], verification_passed=True)
    res = {
        "file": "a.pdf", "file_type": "pdf", "pages": 1,
        "extracted_chunks": [chunk], "pii_outputs": [output],
        "redaction_status": status, "redaction_count": 2, "verification_passed": True,
        "unmapped_findings": [], "inspection_issues": [], "inspection_warnings": [],
    }
    res.update(extra)
    return res


def _states(res):
    return [step.state for step in build_trail(res)]


def test_verified_run_passes_every_step():
    assert _states(_res("VERIFIED_NO_PII_FOUND")) == ["pass"] * 5


def test_rescan_problem_fails_the_rescan_and_verdict():
    res = _res("BLOCKED", inspection_issues=[
        {"page_number": 1, "reason": "Rescan contains low-confidence OCR words", "count": 4},
        {"page_number": 2, "reason": "Rescan contains low-confidence OCR words", "count": 3},
    ])
    steps = build_trail(res)
    assert [s.state for s in steps][3:] == ["fail", "fail"]
    assert steps[3].facts == [(
        "On the redacted copy, some scanned words were hard to read (often next to the black boxes)"
        " (7 words; pages 1, 2)"
    )]


def test_unmapped_finding_stops_at_redaction():
    res = _res("BLOCKED", unmapped_findings=[{"page_number": 1, "entity_type": "PERSON"}])
    assert _states(res)[2:] == ["fail", "skip", "fail"]


def test_skipped_images_and_unread_ink_warn():
    chunk = ExtractionInput(file_id="d", file_name="a.pdf", raw_text="x", images_skipped_count=1,
                            unread_regions=[UnreadRegion(page=1, x=0, y=0, w=5, h=5)])
    res = _res("REVIEW_REQUIRED", extracted_chunks=[chunk],
               inspection_warnings=[{"page_number": 1, "reason": "Images were skipped during rescan",
                                     "count": 1}])
    assert _states(res) == ["warn", "pass", "pass", "warn", "warn"]


def test_failed_rescan_without_page_is_explained():
    res = _res("BLOCKED", verification_passed=False, inspection_issues=[
        {"reason": "Fresh extraction or PII rescan failed", "error": "boom"}])
    blocking, warns = attention(res)
    assert blocking == ["The redacted copy could not be read again: boom"]
    assert warns == []
    assert _states(res)[3] == "fail"


def test_many_pages_are_counted_not_listed():
    items = [{"page_number": p, "reason": "OCR contains low-confidence words", "count": 1}
             for p in range(1, 21)]
    assert summarize(items) == ["Some scanned words were hard to read (20 words; 20 pages)"]


def test_components_escape_values():
    res = _res("BLOCKED", file="<script>.pdf", secs=1.5)
    header = components.case_header(res, "UNSAFE")
    assert "<script>" not in header and "&lt;script&gt;" in header
    assert "Held for review" in components.stamp("BLOCKED")
    finding = res["pii_outputs"][0].detected_entities[0]
    text = components.annotated_text("Contact Helena Vorstermann\n\nauthor: X", [finding], 26)
    assert "Helena" not in text and "PERSON_001" in text and "Hidden content" in text
