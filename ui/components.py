from html import escape

from ui.page_view import entity_color

STAMPS = {
    "VERIFIED_NO_PII_FOUND": ("verified", "Cleared", "Rescan found no PII"),
    "REVIEW_REQUIRED": ("review", "Review", "Some content not fully read"),
    "BLOCKED": ("held", "Held for review", "Manual check before release"),
}
RELEASE_LABELS = {"SAFE": "Release", "UNKNOWN": "Review first", "UNSAFE": "Hold"}
MARKS = {"pass": "✓", "warn": "!", "fail": "!", "skip": "–", "done": "✓", "pending": "",
         "failed": "!", "active": '<span class="spinner"></span>'}


def _type_label(entity_type: str | None) -> str:
    return (entity_type or "").replace("_", " ").title()


def masthead() -> str:
    return (
        '<div class="masthead">'
        '<div class="brand">PII <span class="brand-bar">SENTINEL</span></div>'
        '<div class="brand-tag">Pre-LLM privacy gateway · reads every page, redacts it, '
        "then reads the result again</div></div>"
    )


def stamp(status: str | None) -> str:
    kind, title, sub = STAMPS.get(status, ("review", status or "Unknown", ""))
    return (
        f'<div class="stamp-wrap"><div class="stamp stamp-{kind}">'
        f'<div class="stamp-title">{escape(title)}</div>'
        f'<div class="stamp-sub">{escape(sub)}</div></div></div>'
    )


def token_bar(token: str, entity_type: str | None = None) -> str:
    label = token.strip("[]")
    return (
        f'<span class="token-bar" title="{escape(_type_label(entity_type))}">'
        f'<span class="token-dot" style="background:{entity_color(entity_type)}"></span>'
        f"{escape(label)}</span>"
    )


def case_header(res: dict, release: str) -> str:
    chunks = res.get("extracted_chunks") or []
    case_id = chunks[0].file_id if chunks else "n/a"
    meta = [
        ("File ID", case_id),
        ("Type", res.get("file_type", "").upper()),
        ("Pages", res.get("pages", 0)),
        ("Processed in", f"{res.get('secs', 0)} s"),
    ]
    cells = "".join(
        f'<div class="case-meta"><span>{escape(k)}</span><b>{escape(str(v))}</b></div>' for k, v in meta
    )
    return (
        '<div class="case-file"><div class="case-tab">Analysis report</div>'
        f'<div class="case-title">{escape(res.get("file", ""))}</div>'
        f'<div class="case-row">{cells}'
        f'<div class="case-meta"><span>Release decision</span>'
        f'<b class="release release-{escape(release.lower())}">'
        f"{escape(RELEASE_LABELS.get(release, release))}</b></div>"
        "</div></div>"
    )


def _trail_item(index: int, title: str, state: str, summary: str = "", facts=()) -> str:
    fact_html = "".join(f"<li>{escape(f)}</li>" for f in facts if f)
    return (
        f'<li class="trail-step is-{state}">'
        f'<div class="trail-mark">{MARKS.get(state, "·")}</div>'
        f'<div class="trail-body"><div class="trail-title"><span>{index:02d}</span>{escape(title)}</div>'
        + (f'<div class="trail-summary">{escape(summary)}</div>' if summary else "")
        + (f'<ul class="trail-facts">{fact_html}</ul>' if fact_html else "")
        + "</div></li>"
    )


def evidence_trail(steps) -> str:
    items = "".join(
        _trail_item(i, s.title, s.state, s.summary, s.facts) for i, s in enumerate(steps, start=1)
    )
    return f'<ol class="trail">{items}</ol>'


def live_trail(stages: list[str], current: int, failed: bool, elapsed: float, file_name: str) -> str:
    items = []
    for i, label in enumerate(stages):
        if i < current:
            state = "done"
        elif i == current:
            state = "failed" if failed else "active"
        else:
            state = "pending"
        items.append(_trail_item(i + 1, label, state))
    finished = failed or current >= len(stages)
    title = "Analysis failed" if failed else ("Done" if finished else "Analysing")
    if finished:
        clock = f"{elapsed:.0f}"
    else:
        clock = f'<span class="clock" style="animation-delay:-{elapsed:.0f}s"></span>'
    return (
        f'<div class="live"><div class="live-title">{escape(title)}: '
        f"<b>{escape(file_name)}</b></div>"
        f'<ol class="trail compact">{"".join(items)}</ol>'
        f'<div class="live-time">{clock} s elapsed</div></div>'
    )


def coverage_meter(coverage: dict, required_pct: float, uncertainty: dict) -> str:
    pct = float(coverage.get("coverage_pct", 100.0))
    ok = pct >= required_pct
    parts = [
        f"{uncertainty.get('low', 0):,} low-confidence OCR words",
        f"{uncertainty.get('unread', 0)} unread regions",
        f"{uncertainty.get('skipped', 0)} skipped images",
    ]
    return (
        '<div class="meter">'
        f'<div class="meter-head"><span>Read reliably</span><b>{pct:.1f}%</b></div>'
        f'<div class="meter-track"><div class="meter-fill {"ok" if ok else "low"}" '
        f'style="width:{max(0.0, min(pct, 100.0)):.1f}%"></div>'
        f'<div class="meter-tick" style="left:{required_pct:.0f}%" '
        f'title="{required_pct:.0f}% required"></div></div>'
        f'<div class="meter-note">{coverage.get("reliable", 0):,} of {coverage.get("total", 0):,} '
        f'items (words and unread regions) · {escape(" · ".join(parts))}</div></div>'
    )


def ledger(items, columns: int | None = None) -> str:
    cells = "".join(
        f'<div class="ledger-item"><div class="ledger-num">{escape(str(v))}</div>'
        f'<div class="ledger-label">{escape(label)}</div></div>'
        for v, label in items
    )
    extra = " two-col" if columns == 2 else ""
    return f'<div class="ledger{extra}">{cells}</div>'


def note_list(title: str, lines: list[str], kind: str) -> str:
    if not lines:
        return ""
    items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    return f'<div class="notes notes-{kind}"><div class="notes-title">{escape(title)}</div><ul>{items}</ul></div>'


def checks_panel() -> str:
    steps = [
        ("Read every page", ("Page text with word positions, text inside images (OCR), and "
                             "hidden content such as metadata and comments.")),
        ("Find personal data", "Names, emails, phone numbers, IDs, addresses and dates of birth."),
        ("Redact a copy", ("Each value is blacked out in a copy of your file; the original "
                           "is never changed.")),
        ("Read the copy again", ("The redacted copy is checked from scratch before it is "
                                 "cleared for release.")),
    ]
    rows = "".join(
        f'<div class="check"><div class="check-num">{i:02d}</div><div><b>{escape(t)}</b>'
        f"<p>{escape(d)}</p></div></div>"
        for i, (t, d) in enumerate(steps, start=1)
    )
    return f'<div class="checks"><div class="checks-title">How a document is analysed</div>{rows}</div>'


def intake_heading() -> str:
    return (
        '<div class="intake">'
        '<h1>Redact it <span class="redacted-word">before</span> it reaches an AI.</h1>'
        "<p>Upload a PDF, Word or PowerPoint file. Personal data is found, blacked out in a "
        "copy of the original, and the copy is checked again before you can release it.</p></div>"
    )


def legend() -> str:
    return (
        '<div class="legend">'
        '<span><i class="sw sw-finding"></i>Finding (colour = type)</span>'
        '<span><i class="sw sw-uncertain"></i>OCR word below 60% confidence</span>'
        '<span><i class="sw sw-unread"></i>Ink OCR could not read</span></div>'
    )


def finding_rows(findings) -> str:
    if not findings:
        return '<div class="empty-note">No findings on this page.</div>'
    rows = "".join(
        f'<div class="finding-row">{token_bar(f.token, f.entity_type)}'
        f"<span>{escape(_type_label(f.entity_type))}</span>"
        f'<span class="mono">{f.score:.2f}</span></div>'
        for f in sorted(findings, key=lambda f: f.start)
    )
    return f'<div class="finding-list">{rows}</div>'


def annotated_text(text: str, findings, visible_length: int) -> str:
    def render(start: int, end: int) -> str:
        pieces, pos = [], start
        for f in sorted(findings, key=lambda f: f.start):
            if f.start < pos or f.start >= end:
                continue
            pieces.append(escape(text[pos : f.start]))
            pieces.append(token_bar(f.token, f.entity_type))
            pos = f.end
        pieces.append(escape(text[pos:end]))
        return "".join(pieces).replace("\n", "<br>")

    visible = render(0, min(visible_length, len(text)))
    html_out = f'<div class="paper-page">{visible or "<i>(no visible text)</i>"}'
    if visible_length < len(text):
        html_out += (
            '<div class="hidden-divider">Hidden content (not shown on the page)</div>'
            + render(visible_length, len(text))
        )
    return html_out + "</div>"


def _pct(value) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def accuracy_heading(report: dict, total: dict) -> str:
    return (
        '<div class="intake accuracy-head"><h1>How accurate is it?</h1>'
        f"<p>Measured on the {total['documents']} Cadence case-study documents "
        f"({total['pages']} pages) against identifiers and names the team checked by hand. "
        f"Every number below is a count from the full pipeline: reading, OCR and detection.</p>"
        f'<div class="accuracy-stamp">Measured {escape(report.get("generated_at", ""))}</div></div>'
    )


def accuracy_table(documents: list[dict]) -> str:
    def row(label: str, values: list[str], hint: str = "") -> str:
        cells = "".join(f"<td>{escape(v)}</td>" for v in values)
        hint_html = f'<div class="acc-hint">{escape(hint)}</div>' if hint else ""
        return f"<tr><th>{escape(label)}{hint_html}</th>{cells}</tr>"

    head = "".join(
        f'<th><b>{escape(d["format"])}</b><span>{d["pages"]} pages</span></th>' for d in documents
    )
    det = [d["detection"] for d in documents]
    words = [d["words"] for d in documents]
    rows = [
        row("Precision", [_pct(d["precision"]) for d in det],
            "Share of findings that really are personal data"),
        row("Identifiers found", [
            f'{_pct(d["identifier_recall"])} ({d["identifiers"] - d["identifiers_missed_extraction"] - d["identifiers_missed_detection"]}/{d["identifiers"]})'
            if d["identifiers"] else "none labelled" for d in det],
            "Emails, phones, employee IDs, SSNs, passports..."),
        row("Names covered", [f'{_pct(d["name_recall"])} ({d["names_covered"]}/{d["names"]})' for d in det],
            "Every occurrence of every labelled full name"),
        row("Word-level F1", [_pct(w["f1"]) for w in words],
            "Balance of precision and recall, word by word"),
        row("Findings / false positives", [f'{d["findings"]} / {d["false_positives"]}' for d in det]),
        row("Missed: reading / detection", [
            f'{d["identifiers_missed_extraction"]} / {d["identifiers_missed_detection"]}' for d in det],
            "Where a missed identifier was lost"),
        row("Processing time", [
            f'{d["seconds"]["extraction"] + d["seconds"]["detection"]:.0f} s' for d in documents],
            "Reading + detection (redaction and rescan not included)"),
    ]
    return f'<table class="acc-table"><thead><tr><th></th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def accuracy_notes(report: dict, total: dict) -> str:
    missed, at_reading = total["identifiers_missed"], total["identifiers_missed_extraction"]
    found = [
        f"{total['findings']:,} findings, of which {total['false_positives']} were not personal data.",
        (f"{total['identifiers'] - missed} of {total['identifiers']} labelled identifiers found; "
         f"{at_reading} of the {missed} misses happened while reading the scanned page (OCR "
         "spacing, text wrapped across table cells, a value printed under a barcode), not in "
         "detection."),
        f"{total['names_covered']} of {total['names']} occurrences of labelled full names covered.",
    ]
    method = [
        ("Identifiers (emails, phones, IDs...) were listed by hand per page; one counts as found "
         "only if a finding covers it on that page."),
        "A name or address finding counts as correct only if it contains a hand-checked name word.",
        "The labels are used only to score; detection never reads them.",
        f"Regenerate with: python -m analytics.accuracy (ground truth: {report.get('ground_truth', '')}).",
    ]
    limits = [
        ("These are the same three documents the team used while building the system, so expect "
         "lower scores on documents it has never seen."),
        "All names and identifiers in the case-study files are synthetic.",
        "Recall can only count what was labelled: a value nobody labelled cannot show up as a miss.",
    ]
    return (
        '<div class="acc-notes">'
        + note_list("What the numbers say", found, "info")
        + note_list("How it was measured", method, "info")
        + note_list("Limits of this test", limits, "warn")
        + "</div>"
    )
