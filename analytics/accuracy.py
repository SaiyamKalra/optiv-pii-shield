import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = Path(__file__).with_name("accuracy_report.json")
GROUND_TRUTH = REPO_ROOT / "tests" / "fixtures" / "cadence_ground_truth.json"
DEFAULT_DOCS_DIR = REPO_ROOT.parent / "Case Study 2 Supporting Material"

EXTRACTION_FIELDS = (
    "identifiers_found", "identifier_recall", "rows_intact", "row_integrity", "text_found",
    "images_ocr", "images_skipped", "images_decorative", "low_confidence_words",
    "pages_with_low_confidence", "unread_regions", "extraction_errors",
)


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def score_document(path: Path, truth: dict, pipeline) -> dict:
    from extraction.evaluate import evaluate_file
    from extraction.normalizer import extract_document
    from pii.evaluate import score, word_matrix
    from redaction.office_pipeline import scan_chunks

    started = time.perf_counter()
    chunks = extract_document(path)
    extraction_secs = time.perf_counter() - started
    extraction = evaluate_file(path, truth, chunks=chunks, elapsed=extraction_secs)

    detect_chunks = chunks if path.suffix.lower() == ".pdf" else scan_chunks(chunks)
    started = time.perf_counter()
    results = pipeline.process_document_chunks(detect_chunks)
    detection_secs = time.perf_counter() - started

    s = score(detect_chunks, results, truth)
    w = word_matrix(detect_chunks, results, truth)
    n_ids = s["identifiers"]
    return {
        "file": path.name,
        "format": path.suffix.lstrip(".").upper(),
        "pages": len(chunks),
        "seconds": {"extraction": round(extraction_secs, 1), "detection": round(detection_secs, 1)},
        "extraction": {k: extraction[k] for k in EXTRACTION_FIELDS if k in extraction},
        "detection": {
            "findings": s["findings"],
            "true_positives": s["tp"],
            "false_positives": s["fp"],
            "precision": round(s["precision"], 4),
            "identifiers": n_ids,
            "identifier_recall": round(s["identifier_recall"], 4) if n_ids else None,
            "identifiers_missed_extraction": sum(1 for *_, why in s["missed"] if why == "extraction"),
            "identifiers_missed_detection": sum(1 for *_, why in s["missed"] if why == "detection"),
            "identifiers_partial": len(s["partial"]),
            "names": s["names"],
            "names_covered": s["names_covered"],
            "name_recall": _rate(s["names_covered"], s["names"]),
            "by_type": dict(s["by_type"].most_common()),
            "false_positives_by_type": dict(s["fp_types"].most_common()),
        },
        "words": {
            "tp": w["tp"], "fp": w["fp"], "fn": w["fn"], "tn": w["tn"],
            "precision": round(w["precision"], 4),
            "recall": round(w["recall"], 4),
            "f1": round(w["f1"], 4),
        },
    }


def build_report(docs_dir: Path, ground_truth: Path = GROUND_TRUTH) -> dict:
    from pii.pipeline import PIIPipeline

    truth = json.loads(ground_truth.read_text(encoding="utf-8"))["files"]
    pipeline = PIIPipeline()
    documents, skipped = [], []
    for name, expected in truth.items():
        path = docs_dir / name
        if path.is_file():
            documents.append(score_document(path, expected, pipeline))
        else:
            skipped.append(name)
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "ground_truth": str(ground_truth.relative_to(REPO_ROOT)).replace("\\", "/"),
        "documents": documents,
        "skipped": skipped,
    }


def summarize_report(report: dict) -> dict:
    docs = report.get("documents") or []
    det = [d["detection"] for d in docs]
    words = [d["words"] for d in docs]
    findings = sum(d["findings"] for d in det)
    tp = sum(d["true_positives"] for d in det)
    ids = sum(d["identifiers"] for d in det)
    ids_missed = sum(d["identifiers_missed_extraction"] + d["identifiers_missed_detection"] for d in det)
    names = sum(d["names"] for d in det)
    w_tp, w_fp, w_fn = (sum(w[k] for w in words) for k in ("tp", "fp", "fn"))
    return {
        "documents": len(docs),
        "pages": sum(d["pages"] for d in docs),
        "findings": findings,
        "false_positives": sum(d["false_positives"] for d in det),
        "precision": _rate(tp, findings),
        "identifiers": ids,
        "identifiers_missed": ids_missed,
        "identifiers_missed_extraction": sum(d["identifiers_missed_extraction"] for d in det),
        "identifier_recall": _rate(ids - ids_missed, ids),
        "names": names,
        "names_covered": sum(d["names_covered"] for d in det),
        "name_recall": _rate(sum(d["names_covered"] for d in det), names),
        "word_precision": _rate(w_tp, w_tp + w_fp),
        "word_recall": _rate(w_tp, w_tp + w_fn),
        "word_f1": _rate(2 * w_tp, 2 * w_tp + w_fp + w_fn),
    }


def load_report(path: Path = REPORT_PATH) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description="Score the pipeline against the ground truth.")
    parser.add_argument("docs_dir", nargs="?", default=str(DEFAULT_DOCS_DIR))
    parser.add_argument("--out", default=str(REPORT_PATH))
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    report = build_report(Path(args.docs_dir))
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    for doc in report["documents"]:
        d, w = doc["detection"], doc["words"]
        print(f"{doc['file']}: precision {d['precision']:.1%}, identifier recall "
              f"{d['identifier_recall']}, names {d['names_covered']}/{d['names']}, word F1 {w['f1']:.1%}")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
