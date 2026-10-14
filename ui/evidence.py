from collections import Counter
from dataclasses import dataclass, field

STATUS_LABEL = {
    "VERIFIED_NO_PII_FOUND": "Cleared: the rescan of the redacted copy found no PII.",
    "REVIEW_REQUIRED": "Review: the rescan is clean, but some content could not be fully read.",
    "BLOCKED": "Held for review: a person should check the redacted copy before it is released.",
}

FRIENDLY_REASONS = {
    "OCR contains low-confidence words": "Some scanned words were hard to read",
    "Rescan contains low-confidence OCR words":
        "On the redacted copy, some scanned words were hard to read (often next to the black boxes)",
    "OCR confidence is unavailable": "Some scanned text has no reading-confidence score",
    "Rescan OCR confidence is unavailable": "Some scanned text on the redacted copy has no confidence score",
    "Some images were skipped during extraction": "Some images could not be read (too small or unsupported)",
    "Images were skipped during rescan": "Some images on the redacted copy could not be read",
    "One or more PII findings cannot be mapped to PDF locations":
        "Some findings could not be located on the page",
    "Fresh extraction or PII rescan failed": "The redacted copy could not be read again",
    "Extraction does not cover every PDF page": "Not every page could be read",
    "Fresh extraction did not cover every output page": "Not every page of the redacted copy could be read",
}


@dataclass
class Step:
    title: str
    state: str
    summary: str
    facts: list[str] = field(default_factory=list)


def _is_rescan(item: dict) -> bool:
    text = f"{item.get('reason', '')} {item.get('error', '')}".lower()
    return "rescan" in text or "fresh extraction" in text


def _pages_text(pages) -> str:
    pages = sorted(set(pages))
    if not pages:
        return ""
    if len(pages) > 8:
        return f"{len(pages)} pages"
    return ("page " if len(pages) == 1 else "pages ") + ", ".join(map(str, pages))


def summarize(items: list[dict]) -> list[str]:
    groups: dict[str, dict] = {}
    for item in items or []:
        reason = item.get("reason") or "Unknown issue"
        reason = FRIENDLY_REASONS.get(reason, reason)
        group = groups.setdefault(reason, {"pages": [], "count": 0, "errors": [], "findings": 0})
        if item.get("page_number") is not None:
            group["pages"].append(item["page_number"])
        group["count"] += int(item.get("count") or 0)
        if item.get("error"):
            group["errors"].append(str(item["error"]))
        group["errors"] += [str(e) for e in item.get("errors") or []]
        group["findings"] += len(item.get("findings") or [])
    lines = []
    for reason, group in groups.items():
        parts = []
        if group["count"]:
            unit = " words" if "words" in reason else " images" if "image" in reason else ""
            parts.append(f"{group['count']:,}{unit}")
        if group["findings"]:
            parts.append(f"{group['findings']} finding(s)")
        where = _pages_text(group["pages"])
        if where:
            parts.append(where)
        line = reason + (f" ({'; '.join(parts)})" if parts else "")
        if group["errors"]:
            line += f": {group['errors'][0]}"
        lines.append(line)
    return lines


def attention(res: dict) -> tuple[list[str], list[str]]:
    return summarize(res.get("inspection_issues")), summarize(res.get("inspection_warnings"))


def build_trail(res: dict) -> list[Step]:
    chunks = res.get("extracted_chunks") or []
    outputs = res.get("pii_outputs") or []
    issues = res.get("inspection_issues") or []
    warnings = res.get("inspection_warnings") or []
    status = res.get("redaction_status")
    is_pdf = res.get("file_type", "pdf") == "pdf"

    pages = res.get("pages") or len(chunks)
    ocr_pages = sum(1 for c in chunks if c.ocr_used or c.extraction_method in ("ocr", "hybrid"))
    images = sum(c.images_ocr_count for c in chunks)
    skipped = sum(c.images_skipped_count for c in chunks)
    decorative = sum(c.images_decorative_count for c in chunks)
    hidden = sum(len(c.hidden_content) for c in chunks)
    low = sum(c.low_confidence_word_count for c in chunks)
    unread = sum(len(c.unread_regions) for c in chunks)
    broken = sum(1 for c in chunks if c.extraction_errors)
    image_fact = f"{images} image(s) read with OCR"
    if skipped or decorative:
        image_fact += f", {skipped} skipped, {decorative} known logo(s)"
    facts = [
        f"{pages} page(s), {ocr_pages} needing OCR",
        image_fact,
        f"{hidden} hidden item(s) found (metadata, comments, links...)",
        f"{low:,} low-confidence OCR word(s), {unread} unread region(s)",
    ]
    if broken:
        facts.append(f"{broken} page(s) could not be fully inspected")
    extracted = Step(
        "Extracted",
        "fail" if broken else "warn" if (skipped or low or unread) else "pass",
        "Text, word positions and text inside images were read from the file.",
        facts,
    )

    findings = [e for o in outputs for e in o.detected_entities]
    by_type = Counter(e.entity_type for e in findings)
    detected = Step(
        "Detected",
        "pass",
        f"{len(findings)} finding(s), {len({e.token for e in findings})} distinct value(s).",
        [", ".join(f"{t.replace('_', ' ').title()} {n}" for t, n in by_type.most_common(5))]
        if findings else ["Nothing to redact was found."],
    )

    unmapped = res.get("unmapped_findings") or []
    count = res.get("redaction_count", 0)
    if unmapped:
        redacted = Step(
            "Redacted",
            "fail",
            f"{len(unmapped)} finding(s) could not be placed on the page, so redaction stopped.",
        )
    else:
        how = (
            "Black boxes drawn over each word and the text under them removed; "
            "metadata, comments and links scrubbed."
            if is_pdf else
            "Values masked in every part of the file, including text inside images."
        )
        redacted = Step("Redacted", "pass", how, [f"{count} redaction(s) applied"])

    rescan_issues = [i for i in issues if _is_rescan(i)]
    rescan_warnings = [w for w in warnings if _is_rescan(w)]
    verified_scan = res.get("verification_passed")
    if unmapped:
        rescanned = Step("Rescanned", "skip", "Not run, because redaction was stopped.")
    elif rescan_issues:
        rescanned = Step(
            "Rescanned", "fail",
            "The redacted copy was read again from scratch and raised points to check.",
            summarize(rescan_issues),
        )
    elif verified_scan is False:
        rescanned = Step("Rescanned", "fail",
                         "The rescan still detected possible PII in the redacted copy.")
    elif rescan_warnings:
        rescanned = Step(
            "Rescanned", "warn",
            "The redacted file was read again: no PII found, with some parts not fully readable.",
            summarize(rescan_warnings),
        )
    else:
        rescanned = Step(
            "Rescanned", "pass",
            "The redacted file was read again from scratch and scanned: no PII found.",
        )

    verdict_state = {"VERIFIED_NO_PII_FOUND": "pass", "REVIEW_REQUIRED": "warn"}.get(status, "fail")
    verdict = Step("Verdict", verdict_state, STATUS_LABEL.get(status, str(status)))

    return [extracted, detected, redacted, rescanned, verdict]
