# extraction/evaluate.py
"""Measure extraction quality on the case-study files against a ground-truth file.

python -m extraction.evaluate [DOCS_DIR] [--ground-truth FILE] [--json REPORT]

Reports, per file:
- identifier recall: known identifiers found exactly on their page;
- row integrity: known table rows whose cells come out in order on one line
  (a proxy for structure retention);
- OCR quality, unread regions and hidden content found.
The report contains counts only, never extracted text.
"""

import argparse
import json
import logging
import time
from pathlib import Path

from extraction.normalizer import extract_document

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GROUND_TRUTH = REPO_ROOT / "tests" / "fixtures" / "cadence_ground_truth.json"
DEFAULT_DOCS_DIR = REPO_ROOT.parent / "Case Study 2 Supporting Material"


def _row_on_one_line(text: str, cells: list[str]) -> bool:
    for line in text.splitlines():
        pos = 0
        for cell in cells:
            pos = line.find(cell, pos)
            if pos < 0:
                break
            pos += len(cell)
        else:
            return True
    return False


def evaluate_file(path: Path, truth: dict) -> dict:
    started = time.perf_counter()
    chunks = extract_document(path)
    elapsed = time.perf_counter() - started
    pages = {c.page_number: c.raw_text for c in chunks}

    report = {"file": path.name, "pages": len(chunks), "seconds": round(elapsed, 1)}

    if "identifiers" in truth:
        missed = [
            [page, value] for page, value in truth["identifiers"]
            if value not in pages.get(page, "")
        ]
        total = len(truth["identifiers"])
        report["identifier_recall"] = round((total - len(missed)) / total, 4)
        report["identifiers_found"] = f"{total - len(missed)}/{total}"
        report["identifiers_missed"] = missed

    if "rows" in truth:
        broken = [
            [page, cells] for page, cells in truth["rows"]
            if not _row_on_one_line(pages.get(page, ""), cells)
        ]
        total = len(truth["rows"])
        report["row_integrity"] = round((total - len(broken)) / total, 4)
        report["rows_intact"] = f"{total - len(broken)}/{total}"
        report["rows_broken"] = broken

    if "text" in truth:
        everything = "\n".join(pages.values())
        missed = [t for t in truth["text"] if t not in everything]
        report["text_found"] = f"{len(truth['text']) - len(missed)}/{len(truth['text'])}"
        report["text_missed"] = missed

    report["images_ocr"] = sum(c.images_ocr_count for c in chunks)
    report["images_skipped"] = sum(c.images_skipped_count for c in chunks)
    report["images_decorative"] = sum(getattr(c, "images_decorative_count", 0) for c in chunks)
    report["low_confidence_words"] = sum(c.low_confidence_word_count for c in chunks)
    report["pages_with_low_confidence"] = sum(c.low_confidence_word_count > 0 for c in chunks)
    report["unread_regions"] = sum(len(getattr(c, "unread_regions", []) or []) for c in chunks)
    hidden: dict[str, int] = {}
    for c in chunks:
        for item in getattr(c, "hidden_content", []) or []:
            hidden[item.kind] = hidden.get(item.kind, 0) + 1
    report["hidden_content"] = hidden
    report["extraction_errors"] = sum(len(getattr(c, "extraction_errors", []) or []) for c in chunks)
    if "min_images_ocr" in truth:
        report["min_images_ocr_met"] = report["images_ocr"] >= truth["min_images_ocr"]
    return report


def evaluate(docs_dir: Path, ground_truth: Path) -> list[dict]:
    truth = json.loads(ground_truth.read_text(encoding="utf-8"))["files"]
    reports = []
    for name, expected in truth.items():
        path = docs_dir / name
        if path.is_file():
            reports.append(evaluate_file(path, expected))
        else:
            reports.append({"file": name, "skipped": "file not found"})
    return reports

def evaluate_accuracy(ground_truth_entities: list, detected_entities: list) -> dict:
    """
    ground_truth_entities: [(start, end, entity_type), ...]
    detected_entities: [(start, end, entity_type), ...]
    """
    gt_set = set(ground_truth_entities)
    pred_set = set(detected_entities)

    true_positives = len(gt_set & pred_set)
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "true_positives (correct)": true_positives,
        "false_positives (over-redacted)": false_positives,
        "false_negatives (missed PII)": false_negatives,
        "precision_pct": round(precision * 100, 2),
        "recall_pct": round(recall * 100, 2),
        "f1_score_pct": round(f1_score * 100, 2),
    }
    
def main():
    parser = argparse.ArgumentParser(description="Measure extraction quality on known documents.")
    parser.add_argument("docs_dir", nargs="?", default=str(DEFAULT_DOCS_DIR))
    parser.add_argument("--ground-truth", default=str(DEFAULT_GROUND_TRUTH))
    parser.add_argument("--json", metavar="FILE", help="Also save the report as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    reports = evaluate(Path(args.docs_dir), Path(args.ground_truth))
    for r in reports:
        print(f"\n== {r['file']}")
        for key, value in r.items():
            if key != "file":
                print(f"  {key}: {value}")
    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

