"""Render the case-file screens with a prepared result (no OCR or upload needed)."""

from pathlib import Path

import pandas as pd
import pymupdf
import pytest
from streamlit.testing.v1 import AppTest

from analytics.metrics import reading_coverage
from models.schemas import BoundingBox, DetectedEntityDetail, ExtractionInput, PIIOutput

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _result(status: str) -> dict:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Contact Helena Vorstermann today")
    words = page.get_text("words")
    data = doc.tobytes()
    doc.close()

    text, boxes = "", []
    for x0, y0, x1, y1, word, *_ in words:
        start = len(text) + (1 if text else 0)
        text = f"{text} {word}" if text else word
        boxes.append(BoundingBox(page=1, text=word, x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                                 char_start=start, char_end=start + len(word)))
    chunk = ExtractionInput(file_id="doc_test", file_name="original.pdf", raw_text=text,
                            bounding_boxes=boxes)
    start = text.index("Helena")
    finding = DetectedEntityDetail(token="[PERSON_001]", entity_type="PERSON",
                                   original_value="Helena Vorstermann", start=start,
                                   end=start + 18, score=0.9)
    output = PIIOutput(file_id="doc_test", file_name="original.pdf", status="SUCCESS",
                       sanitized_text=text.replace("Helena Vorstermann", "[PERSON_001]"),
                       detected_entities=[finding], verification_passed=True)
    coverage = reading_coverage([chunk])
    findings = pd.DataFrame([{
        "ID": "F-00001", "Page": 1, "Source": "body", "Type": "PERSON",
        "Detector": "spaCy NER + name filter", "Confidence": 0.9, "Token": "[PERSON_001]",
        "Context": "Detected PERSON",
    }])
    return {
        "file": "case.pdf", "file_type": "pdf", "pages": 1, "ocr_pages": 0,
        "findings": findings, "original_bytes": data, "sanitized_bytes": data,
        "residual": 0, "secs": 1.2, "redaction_status": status, "redaction_count": 2,
        "verification_passed": status != "BLOCKED", "unmapped_findings": [],
        "inspection_issues": [] if status != "BLOCKED" else [
            {"page_number": 1, "reason": "Rescan contains low-confidence OCR words", "count": 2}],
        "inspection_warnings": [], "extracted_chunks": [chunk], "pii_outputs": [output],
        "total_words": 4, "reliable_words": coverage["reliable"],
        "coverage_total": coverage["total"], "coverage": coverage,
        "markdown": "---\ntitle: case.pdf\n---\n",  # skip loading the optimizer's model
    }


def test_intake_screen_renders():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any(b.label == "Analyse document" for b in at.button)


@pytest.mark.parametrize("status", ["VERIFIED_NO_PII_FOUND", "BLOCKED"])
def test_case_file_renders_every_tab(status):
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["res"] = _result(status)
    at.session_state["screen"] = "case"
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Verdict", "Document", "Findings", "Analytics"]
    page_html = " ".join(m.value for m in at.markdown)
    assert ("Held for review" if status == "BLOCKED" else "Cleared") in page_html
    assert "Helena" not in page_html  # findings are shown as tokens only
