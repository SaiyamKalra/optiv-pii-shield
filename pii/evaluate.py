"""Measure PII detection quality on saved extraction chunks (no OCR, ~10 s for the PDF).

    python -m extraction FILE.pdf --save-chunks %TEMP%/chunks.json    # once
    python -m pii.evaluate %TEMP%/chunks.json [--top 20]

Scored against tests/fixtures/cadence_ground_truth.json for that file:
- structured identifiers (emails, phones, IDs...) count as true positives, except
  policy numbers and exception IDs, which are not personal data;
- a PERSON / ORGANIZATION / LOCATION / address finding is a true positive only if it
  contains a word from the file's hand-checked `people` list, otherwise false positive;
- misses: each known identifier (missed, or only partly covered), split into
  "extraction" (not in the extracted text) and "detection", plus every occurrence
  of each listed full name (name recall).

Output contains PII values from the sample files; keep it out of the repo.
"""

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

from pydantic import TypeAdapter

from models.schemas import ExtractionInput

GROUND_TRUTH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "cadence_ground_truth.json"
NAME_LIKE_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "NRP", "STREET_ADDRESS"}
NOT_PERSONAL_TYPES = {"DOCUMENT_REFERENCE", "EXCEPTION_ID"}


def _is_true_positive(finding, people: set[str]) -> bool:
    """Structured identifiers are correct; a name or address must hold a labelled word."""
    if finding.entity_type in NOT_PERSONAL_TYPES:
        return False
    if finding.entity_type in NAME_LIKE_TYPES:
        return any(t in people for t in re.findall(r"[A-Za-z][A-Za-z-]*", finding.original_value))
    return True


def _flexible(value: str) -> re.Pattern:
    """Match a value with any (or no) whitespace between its characters' words."""
    return re.compile(r"\s*".join(map(re.escape, value.replace(" ", ""))))


def _covered(text: str, findings: list, start: int, end: int) -> bool:
    return all(
        any(f.start <= i < f.end for f in findings)
        for i in range(start, end)
        if not text[i].isspace()
    )


def score(chunks: list[ExtractionInput], results, truth: dict) -> dict:
    people = set(truth.get("people", []))
    pages = {c.page_number: c.raw_text for c in chunks}
    by_page = {r.page_number: r.detected_entities for r in results}
    all_findings = [e for r in results for e in r.detected_entities]

    tp = fp = 0
    fp_values: Counter = Counter()
    fp_types: Counter = Counter()
    for f in all_findings:
        if _is_true_positive(f, people):
            tp += 1
        else:
            fp += 1
            fp_types[f.entity_type] += 1
            fp_values[" ".join(f.original_value.split())] += 1

    missed, partial = [], []
    for page, value in truth.get("identifiers", []):
        text, found = pages.get(page, ""), by_page.get(page, [])
        spans = [m.span() for m in _flexible(value).finditer(text)]
        if not spans:
            missed.append((page, value, "extraction"))
        elif not any(f.start < b and f.end > a for f in found for a, b in spans):
            missed.append((page, value, "detection"))
        elif not any(_covered(text, found, a, b) for a, b in spans):
            partial.append((page, value))

    name_total = name_hit = 0
    name_misses: Counter = Counter()
    for name in truth.get("full_names", []):
        pattern = _flexible(name)
        for page, text in pages.items():
            for m in pattern.finditer(text):
                name_total += 1
                if _covered(text, by_page.get(page, []), *m.span()):
                    name_hit += 1
                else:
                    name_misses[(page, name)] += 1

    n_ids = len(truth.get("identifiers", []))
    return {
        "findings": len(all_findings),
        "by_type": Counter(f.entity_type for f in all_findings),
        "tp": tp,
        "fp": fp,
        "precision": tp / max(1, len(all_findings)),
        "fp_types": fp_types,
        "fp_values": fp_values,
        "identifiers": n_ids,
        "missed": missed,
        "partial": partial,
        "identifier_recall": (n_ids - len(missed)) / max(1, n_ids),
        "names": name_total,
        "names_covered": name_hit,
        "name_misses": name_misses,
    }


def word_matrix(chunks: list[ExtractionInput], results, truth: dict) -> dict:
    """Confusion matrix over every whitespace-separated word of every page.

    Actual PII word: inside a known identifier occurrence, a word of the file's `people`
    list, or inside a finding that is correct at finding level. Predicted PII word:
    inside any finding that is not a policy number. Names split by OCR or layout still
    count word by word. Precision is therefore bounded by the finding-level labels and
    recall by the identifier and name labels.
    """
    people = set(truth.get("people", []))
    by_page = {r.page_number: r.detected_entities for r in results}
    tp = fp = fn = tn = 0
    for chunk in chunks:
        text, findings = chunk.raw_text, by_page.get(chunk.page_number, [])
        actual_spans = []
        for page, value in truth.get("identifiers", []):
            if page == chunk.page_number:
                actual_spans += [m.span() for m in _flexible(value).finditer(text)]
        # Words inside a finding that is correct at finding level (so a middle initial
        # in "Naomi K. Osadebe" counts); recall below still rests on labels only.
        actual_spans += [(f.start, f.end) for f in findings if _is_true_positive(f, people)]
        predicted = [(f.start, f.end) for f in findings if f.entity_type not in NOT_PERSONAL_TYPES]
        for m in re.finditer(r"\S+", text):
            a, b = m.span()
            word = m.group().strip(".,;:()[]\"'")
            is_pii = (
                any(s < b and e > a for s, e in actual_spans)
                or (word in people and "@" not in m.group())
            )
            is_pred = any(s < b and e > a for s, e in predicted)
            if is_pii and is_pred:
                tp += 1
            elif is_pred:
                fp += 1
            elif is_pii:
                fn += 1
            else:
                tn += 1
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def main():
    parser = argparse.ArgumentParser(description="Score PII detection on saved chunks.")
    parser.add_argument("chunks", help="ExtractionInput JSON from --save-chunks")
    parser.add_argument("--top", type=int, default=20, help="How many false-positive values to list")
    args = parser.parse_args()
    logging.disable(logging.WARNING)

    from pii.pipeline import PIIPipeline

    chunks = TypeAdapter(list[ExtractionInput]).validate_python(
        json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    )
    # DOCX/PPTX carry hidden text (metadata, comments, notes): scan it after the page
    # text, as the Office pipeline does.
    for i, chunk in enumerate(chunks):
        hidden = "\n".join(h.text for h in chunk.hidden_content if h.text)
        if hidden:
            chunks[i] = chunk.model_copy(update={"raw_text": f"{chunk.raw_text}\n\n{hidden}"})
    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))["files"].get(chunks[0].file_name)
    if truth is None:
        raise SystemExit(f"No ground truth for {chunks[0].file_name}")

    import time
    started = time.perf_counter()
    results = PIIPipeline().process_document_chunks(chunks)
    secs = time.perf_counter() - started

    r = score(chunks, results, truth)
    print(f"{chunks[0].file_name}: {len(chunks)} pages, detection {secs:.1f} s")
    print(f"findings {r['findings']}  TP {r['tp']}  FP {r['fp']}  precision {r['precision']:.0%}")
    print(f"by type: {dict(r['by_type'].most_common())}")
    print(f"FP by type: {dict(r['fp_types'].most_common())}")
    print(f"identifiers: {r['identifiers']}  missed {len(r['missed'])}  partial {len(r['partial'])}"
          f"  recall {r['identifier_recall']:.0%}")
    print(f"full-name occurrences covered: {r['names_covered']}/{r['names']}")
    m = word_matrix(chunks, results, truth)
    print(f"word-level matrix: TP {m['tp']}  FP {m['fp']}  FN {m['fn']}  TN {m['tn']}  |  "
          f"precision {m['precision']:.1%}  recall {m['recall']:.1%}  F1 {m['f1']:.1%}")
    print(f"\ntop FP: {r['fp_values'].most_common(args.top)}")
    for page, value, why in r["missed"]:
        print(f"  MISSED p{page} [{why}] {value}")
    for page, value in r["partial"]:
        print(f"  PARTIAL p{page} {value}")
    for (page, name), n in sorted(r["name_misses"].items()):
        print(f"  NAME NOT COVERED p{page} {name}" + (f" x{n}" if n > 1 else ""))


if __name__ == "__main__":
    main()
