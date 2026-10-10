# tests/test_real_documents.py
"""Regression check on the real case-study files: extraction must not get worse.

Slow (about a minute) and needs the sample files, so it only runs when asked:

    RUN_REAL_DOCS=1 pytest tests/test_real_documents.py -v

The files are found in ../Case Study 2 Supporting Material, or CADENCE_DOCS_DIR.
Thresholds are the measured results; raise them when extraction improves.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from extraction.evaluate import DEFAULT_DOCS_DIR, DEFAULT_GROUND_TRUTH, evaluate_file

DOCS_DIR = Path(os.environ.get("CADENCE_DOCS_DIR", DEFAULT_DOCS_DIR))

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_DOCS") != "1", reason="set RUN_REAL_DOCS=1 to run (slow)"
)

POLICY_PDF = "Cadence_Group_Risk_Management_Policy__v6.0.pdf"
ORG_PPTX = "Cadence_Financial_Group_Organizational_Pack.pptx"
TRAINING_DOCX = "Cadence_TPRM_Training.docx"


def _report(name: str) -> dict:
    import json

    path = DOCS_DIR / name
    if not path.is_file():
        pytest.skip(f"{path} not found")
    truth = json.loads(Path(DEFAULT_GROUND_TRUTH).read_text(encoding="utf-8"))["files"][name]
    return evaluate_file(path, truth)


def test_policy_pdf():
    r = _report(POLICY_PDF)
    assert r["pages"] == 35
    assert r["identifier_recall"] >= 0.94, r["identifiers_missed"]
    assert r["row_integrity"] >= 0.875, r["rows_broken"]
    assert r["extraction_errors"] == 0
    assert r["hidden_content"].get("metadata", 0) >= 1
    assert r["low_confidence_words"] <= 300  # baseline before hardening: 535


def test_org_pack_pptx():
    r = _report(ORG_PPTX)
    assert r["pages"] == 9
    assert r["identifier_recall"] == 1.0, r["identifiers_missed"]
    assert r["row_integrity"] == 1.0, r["rows_broken"]
    assert r["images_decorative"] == 9  # the logo on every slide
    assert r["pages_with_low_confidence"] == 0
    assert r["hidden_content"].get("metadata", 0) >= 1


def test_training_docx():
    r = _report(TRAINING_DOCX)
    assert r["text_missed"] == []
    assert r["min_images_ocr_met"]
    assert r["extraction_errors"] == 0
    assert r["hidden_content"].get("metadata", 0) >= 1
