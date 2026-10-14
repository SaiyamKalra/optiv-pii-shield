import json
import re
from pathlib import Path

from analytics.accuracy import REPORT_PATH, load_report, summarize_report
from ui import components


def _doc(fmt, findings, tp, ids, missed_ext, names, covered, w):
    return {
        "file": f"x.{fmt.lower()}", "format": fmt, "pages": 2,
        "seconds": {"extraction": 1.0, "detection": 2.0},
        "extraction": {},
        "detection": {
            "findings": findings, "true_positives": tp, "false_positives": findings - tp,
            "precision": tp / findings, "identifiers": ids,
            "identifier_recall": (ids - missed_ext) / ids if ids else None,
            "identifiers_missed_extraction": missed_ext, "identifiers_missed_detection": 0,
            "identifiers_partial": 0, "names": names, "names_covered": covered,
            "name_recall": covered / names, "by_type": {}, "false_positives_by_type": {},
        },
        "words": {"tp": w[0], "fp": w[1], "fn": w[2], "tn": 100, "precision": 0, "recall": 0, "f1": 0},
    }


def test_totals_are_computed_from_summed_counts():
    report = {"documents": [
        _doc("PDF", 100, 98, 50, 2, 40, 39, (90, 2, 8)),
        _doc("DOCX", 10, 10, 0, 0, 10, 10, (10, 0, 0)),
    ]}
    total = summarize_report(report)
    assert total["precision"] == round(108 / 110, 4)
    assert total["identifier_recall"] == 0.96
    assert total["name_recall"] == round(49 / 50, 4)
    assert total["word_f1"] == round(200 / (200 + 2 + 8), 4)
    assert total["false_positives"] == 2


def test_table_shows_documents_without_identifiers():
    report = {"documents": [_doc("DOCX", 10, 10, 0, 0, 10, 10, (10, 0, 0))]}
    html = components.accuracy_table(report["documents"])
    assert "none labelled" in html and "DOCX" in html


def test_committed_report_holds_no_document_values():
    report = load_report()
    if report is None:
        return
    text = json.dumps(report)
    assert not re.search(r"@|EMP-\d|DIR-\d|\+\d{1,3} ", text)
    truth = json.loads((Path(REPORT_PATH).parents[1] / report["ground_truth"]).read_text(encoding="utf-8"))
    names = {n for f in truth["files"].values() for n in f.get("full_names", [])}
    assert not any(name in text for name in names)
