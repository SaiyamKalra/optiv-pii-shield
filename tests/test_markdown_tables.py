# tests/test_markdown_tables.py
"""Tables in the token-optimised Markdown: rebuilt from OCR word positions (scanned
PDFs) or from "a | b" rows (DOCX/PPTX), with PII replaced by tokens."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from models.schemas import BoundingBox, DetectedEntityDetail, ExtractionInput, PIIOutput
from redaction.token_redaction.markdown_tables import (
    find_table_regions,
    page_lines,
    pipe_rows_to_markdown,
)
from redaction.token_redaction.token_optimizer import TokenOptimizer

SIZE = 8.0  # word height in points


def _chunk(rows: list[list[tuple[float, str]]], gap_y: float = 4.0) -> ExtractionInput:
    """A page of OCR words: rows of (x, word); each row is one raw_text line."""
    text, boxes, y = "", [], 100.0
    for row in rows:
        for i, (x, word) in enumerate(row):
            if i:
                text += " "
            start = len(text)
            text += word
            boxes.append(BoundingBox(page=1, text=word, x=x, y=y, w=len(word) * 4.0, h=SIZE,
                                     char_start=start, char_end=len(text), source="ocr",
                                     confidence=95.0))
        text += "\n"
        y += SIZE + gap_y
    return ExtractionInput(file_id="f", file_name="t.pdf", page_number=1,
                           raw_text=text.rstrip("\n"), bounding_boxes=boxes)


def _output(chunk: ExtractionInput, values: dict[str, str]) -> PIIOutput:
    """Findings for each value (-> token) wherever it occurs in the page text."""
    found = []
    for value, token in values.items():
        start = chunk.raw_text.find(value)
        while start >= 0:
            found.append(DetectedEntityDetail(token=token, entity_type="PERSON", original_value=value,
                                              start=start, end=start + len(value), score=1.0))
            start = chunk.raw_text.find(value, start + 1)
    sanitized = chunk.raw_text
    for f in sorted(found, key=lambda f: f.start, reverse=True):
        sanitized = sanitized[: f.start] + f.token + sanitized[f.end:]
    return PIIOutput(file_id="f", file_name="t.pdf", page_number=1, status="SUCCESS",
                     sanitized_text=sanitized, detected_entities=found, verification_passed=True)


TABLE = [
    [(10, "NAME"), (120, "ROLE"), (240, "OFFICE")],
    [(10, "Naomi"), (34, "Osadebe"), (120, "IAM"), (136, "Engineer"), (240, "New"), (256, "York")],
    [(10, "Tobias"), (38, "Lindqvist"), (120, "Head,"), (144, "Enterprise"), (240, "London")],
    [(120, "Risk"), (140, "Framework")],  # wrapped cell: continues the row above
    [(10, "Helena"), (38, "Vorstermann"), (120, "Chief"), (144, "Privacy"), (240, "London")],
]


def test_ocr_rows_with_aligned_gaps_become_a_table_with_wrapped_cells_merged():
    chunk = _chunk([[(10, "Intro"), (34, "paragraph"), (76, "before"), (104, "it.")]] + TABLE)
    regions = find_table_regions(page_lines(chunk))
    assert len(regions) == 1 and len(regions[0].lines) == 5

    md = TokenOptimizer.__new__(TokenOptimizer).optimize_output([_output(chunk, {})], chunks=[chunk])
    assert "| NAME | ROLE | OFFICE |" in md
    assert "| Tobias Lindqvist | Head, Enterprise Risk Framework | London |" in md
    assert "Intro paragraph before it." in md


def test_prose_with_uneven_spaces_is_not_a_table():
    prose = [
        [(10, "The"), (30, "policy"), (62, "applies"), (98, "to"), (112, "every"), (140, "entity.")],
        [(10, "Owners"), (44, "review"), (80, "each"), (104, "risk"), (126, "quarterly"), (170, "and")],
        [(10, "report"), (42, "breaches"), (86, "to"), (100, "the"), (118, "committee"), (166, "now.")],
    ]
    assert find_table_regions(page_lines(_chunk(prose))) == []


def test_bulleted_list_is_not_a_table():
    rows = [[(10, "•"), (40, "First"), (66, "item")], [(10, "•"), (40, "Second"), (70, "item")],
            [(10, "•"), (40, "Third"), (66, "item")]]
    assert find_table_regions(page_lines(_chunk(rows))) == []


def test_pii_in_cells_becomes_tokens_and_never_appears():
    chunk = _chunk(TABLE)
    out = _output(chunk, {"Naomi Osadebe": "[PERSON_001]", "Tobias Lindqvist": "[PERSON_002]"})
    md = TokenOptimizer.__new__(TokenOptimizer).optimize_output([out], chunks=[chunk])
    assert "| [PERSON_001] | IAM Engineer | New York |" in md
    assert "Naomi" not in md and "Osadebe" not in md and "Lindqvist" not in md


def test_known_value_missed_in_one_place_is_still_tokenized():
    # Detected on page 1; on page 2 the same number wraps and was not detected.
    p1 = PIIOutput(file_id="f", file_name="t.docx", page_number=1, status="SUCCESS",
                   sanitized_text="Call [PHONE_001]", verification_passed=True,
                   detected_entities=[DetectedEntityDetail(token="[PHONE_001]", entity_type="PHONE_NUMBER",
                                                           original_value="+33 15555 0144", start=5, end=19,
                                                           score=1.0)])
    p2 = PIIOutput(file_id="f", file_name="t.docx", page_number=2, status="SUCCESS",
                   sanitized_text="Backup line +33 1 5555\n0144 after hours.", verification_passed=True)
    md = TokenOptimizer.__new__(TokenOptimizer).optimize_output([p1, p2])
    assert "5555" not in md and md.count("[PHONE_001]") == 2


def test_pipe_rows_become_a_table_and_empty_cells_stay_empty():
    parts = pipe_rows_to_markdown("Intro\nType | Owner | Notes\nVendor | Risk team |  |\nOne | line")
    tables = [c for k, c in parts if k == "table"]
    expected = (
        "| Type | Owner | Notes |\n| --- | --- | --- |\n| Vendor | Risk team |  |\n"
        "| One | line |  |"
    )
    assert tables == [expected]
    assert pipe_rows_to_markdown("Just one | line") == [("text", "Just one | line")]


def test_clean_text_keeps_hyphens_inside_words():
    clean = TokenOptimizer.__new__(TokenOptimizer)._clean_text
    assert clean("RSK-4502 owned by Okonkwo-Bryce") == "RSK-4502 owned by Okonkwo-Bryce"
    assert clean("infor-\nmation") == "information"


def test_empty_page_does_not_fall_back_to_raw_text():
    chunk = ExtractionInput(file_id="f", file_name="t.pdf", page_number=1, raw_text="Naomi Osadebe")
    out = PIIOutput(file_id="f", file_name="t.pdf", page_number=1, status="SUCCESS",
                    sanitized_text="", verification_passed=True)
    md = TokenOptimizer.__new__(TokenOptimizer).optimize_output([out], chunks=[chunk])
    assert "Osadebe" not in md
