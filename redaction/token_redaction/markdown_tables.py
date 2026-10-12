"""Rebuild tables for the token-optimised Markdown.

Scanned PDFs have no table structure, only OCR words with positions. A table shows up
as rows (OCR lines) whose words leave the same vertical strips of empty space: the
gaps between columns. This module finds those strips (a projection profile) and turns
the rows into Markdown tables. DOCX/PPTX tables already arrive as "a | b | c" lines.

All thresholds are relative to the page's own text size, so nothing depends on a
particular document. Cells are built from the same findings as the sanitized text:
each detected value becomes its token once, and never appears in plain text.
"""

import bisect
import re
from dataclasses import dataclass, field
from statistics import median

# A gap between two words on a line is a column gap if it is at least this many times
# the page's median word height. Spaces between words are ~0.3x; columns 1.2-6x.
COLUMN_GAP_RATIO = 0.9
# Lines further apart than this many word heights are not in the same table.
MAX_ROW_SPACING_RATIO = 2.0
MIN_TABLE_ROWS = 3
# Share of rows that must have words in at least two columns.
MIN_MULTI_COLUMN_SHARE = 0.7
# A 2-column "table" whose first column is always this short is a bullet/numbered list.
LIST_MARKER_MAX_CHARS = 3


@dataclass
class Word:
    x0: float
    x1: float
    y0: float
    y1: float
    start: int
    end: int

    @property
    def center(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class Line:
    index: int  # line number in the page's raw_text
    words: list[Word] = field(default_factory=list)

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)


@dataclass
class Region:
    lines: list[Line]
    separators: list[tuple[float, float]]


# --- sanitizing -------------------------------------------------------------------------

def sanitize_range(raw_text: str, findings, a: int, b: int, emitted: set) -> str:
    """raw_text[a:b] with every detected value replaced by its token.

    A value's token is written once (the first time any part of it is seen; `emitted`
    tracks that across calls), so a value split over two cells or segments is never
    left in plain text and never duplicated."""
    pieces, pos = [], a
    for f in sorted(findings, key=lambda f: f.start):
        if f.end <= a or f.start >= b:
            continue
        if f.start > pos:
            pieces.append(raw_text[pos : f.start])
        key = (f.start, f.end)
        if key not in emitted:
            pieces.append(f.token)
            emitted.add(key)
        pos = max(pos, min(f.end, b))
    if pos < b:
        pieces.append(raw_text[pos:b])
    return "".join(pieces)


# --- layout analysis --------------------------------------------------------------------

def page_lines(chunk) -> list[Line]:
    """The page's positioned words grouped by the raw_text line they sit on."""
    text = chunk.raw_text or ""
    newlines = [i for i, ch in enumerate(text) if ch == "\n"]
    lines: dict[int, Line] = {}
    for box in chunk.bounding_boxes or []:
        if box.coord_space != "page_pt" or box.char_start is None or box.char_end is None:
            continue
        index = bisect.bisect_left(newlines, box.char_start)
        lines.setdefault(index, Line(index)).words.append(
            Word(box.x, box.x + box.w, box.y, box.y + box.h, box.char_start, box.char_end)
        )
    for line in lines.values():
        line.words.sort(key=lambda w: w.x0)
    return [lines[i] for i in sorted(lines)]


def _column_gaps(line: Line, gap: float) -> int:
    return sum(1 for a, b in zip(line.words, line.words[1:]) if b.x0 - a.x1 >= gap)


def _separators(rows: list[Line], gap: float) -> list[tuple[float, float]]:
    """Interior x-strips at least `gap` wide that (almost) no row covers."""
    lo = min(w.x0 for r in rows for w in r.words)
    hi = max(w.x1 for r in rows for w in r.words)
    width = int(hi - lo) + 1
    cover = [0] * width
    for row in rows:
        row_cover = [False] * width
        for w in row.words:
            for x in range(int(w.x0 - lo), min(width, int(w.x1 - lo) + 1)):
                row_cover[x] = True
        for x, covered in enumerate(row_cover):
            cover[x] += covered
    # One row (a header or a long cell) may cross a column gap in a larger table.
    tolerance = 1 if len(rows) >= 4 else 0
    seps, run_start = [], None
    for x in range(width + 1):
        free = x < width and cover[x] <= tolerance
        if free and run_start is None:
            run_start = x
        elif not free and run_start is not None:
            if run_start > 0 and x < width and x - run_start >= gap:
                seps.append((lo + run_start, lo + x))
            run_start = None
    return seps


def _column_of(x: float, cuts: list[float]) -> int:
    return bisect.bisect_left(cuts, x)


def _cuts(separators):
    return [(a + b) / 2 for a, b in separators]


def _cell_words(row: Line, cuts: list[float]) -> list[list[Word]]:
    cells: list[list[Word]] = [[] for _ in range(len(cuts) + 1)]
    for w in row.words:
        cells[_column_of(w.center, cuts)].append(w)
    return cells


def find_table_regions(lines: list[Line]) -> list[Region]:
    """Runs of rows whose words line up in columns (see module docstring)."""
    words = [w for line in lines for w in line.words]
    if not words:
        return []
    size = median(w.y1 - w.y0 for w in words)
    gap = COLUMN_GAP_RATIO * size
    max_spacing = MAX_ROW_SPACING_RATIO * size

    regions, i = [], 0
    while i < len(lines):
        if _column_gaps(lines[i], gap) == 0:
            i += 1
            continue
        run = [lines[i]]
        for nxt in lines[i + 1 :]:
            if not (0 <= nxt.y0 - run[-1].y1 < max_spacing):
                break
            if not _separators(run + [nxt], gap):
                break
            run.append(nxt)
        # Growth can run past the table into the text below (one row may cross a
        # column gap). Keep the longest leading part of the run that is a table.
        region = None
        for k in range(len(run), MIN_TABLE_ROWS - 1, -1):
            region = _accept(run[:k], gap)
            if region:
                break
        if region:
            regions.append(region)
            i += len(region.lines)
        else:
            i += 1
    return regions


def _crosses(row: Line, seps) -> bool:
    """True if a word spans the middle of a column gap (the line words are assigned
    to columns by), as text running across the table does. A long cell may reach into
    the edge of a gap without crossing it."""
    return any(w.x0 < cut < w.x1 for w in row.words for cut in _cuts(seps))


def _accept(run: list[Line], gap: float) -> Region | None:
    if len(run) < MIN_TABLE_ROWS:
        return None
    seps = _separators(run, gap)
    # A trailing row that runs across a column gap is the paragraph after the table.
    while seps and len(run) > MIN_TABLE_ROWS and _crosses(run[-1], seps):
        run = run[:-1]
        seps = _separators(run, gap)
    if not seps:
        return None
    cuts = _cuts(seps)
    cells = [_cell_words(row, cuts) for row in run]
    multi = sum(1 for row in cells if sum(1 for c in row if c) >= 2)
    if multi < MIN_MULTI_COLUMN_SHARE * len(run):
        return None
    if len(seps) == 1 and all(
        sum(w.end - w.start for w in row[0]) <= LIST_MARKER_MAX_CHARS for row in cells
    ):
        return None  # "• item" / "1. item": a list, not a table
    return Region(run, seps)


# --- Markdown ---------------------------------------------------------------------------

def _escape(cell: str) -> str:
    return " ".join(cell.replace("|", "\\|").split())


def _markdown_table(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(_escape(c) for c in rows[0]) + " |",
             "| " + " | ".join("---" for _ in range(width)) + " |"]
    lines += ["| " + " | ".join(_escape(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def region_to_markdown(region: Region, raw_text: str, findings, emitted: set) -> str:
    """Cells from each word's sanitized text; the first row is the header. A row that
    continues the one above (wrapped cell text) is merged into it: its first column is
    empty, or it fills a single cell while the table has several columns."""
    cuts = _cuts(region.separators)
    rows: list[list[list[Word]]] = []
    for line in region.lines:
        cells = _cell_words(line, cuts)
        filled = sum(1 for c in cells if c)
        if rows and (not cells[0] or (filled == 1 and len(cells) > 2)):
            for column, words in enumerate(cells):
                rows[-1][column].extend(words)
        else:
            rows.append(cells)
    text_rows = [
        [
            " ".join(
                t for w in sorted(cell, key=lambda w: w.start)
                if (t := sanitize_range(raw_text, findings, w.start, w.end, emitted).strip())
            )
            for cell in row
        ]
        for row in rows
    ]
    return _markdown_table(text_rows)


_PIPE_ROW = re.compile(r"\s\|(\s|$)")
_PIPE_SPLIT = re.compile(r"\s*\|\s*")


def pipe_rows_to_markdown(text: str) -> list[tuple[str, str]]:
    """Split text into ("text", ...) and ("table", markdown) parts. A run of two or more
    consecutive "a | b | c" lines (as DOCX/PPTX extraction writes table rows) becomes
    a Markdown table."""
    parts: list[tuple[str, str]] = []
    buffer: list[str] = []
    run: list[list[str]] = []

    def flush_run():
        if len(run) >= 2:
            if buffer:
                parts.append(("text", "\n".join(buffer)))
                buffer.clear()
            parts.append(("table", _markdown_table(run)))
        else:
            buffer.extend(" | ".join(r) for r in run)
        run.clear()

    for line in text.split("\n"):
        if _PIPE_ROW.search(line):
            # Empty cells arrive as "a |  | b" or a trailing " |": split on every bar.
            cells = _PIPE_SPLIT.split(line.strip())
            if cells and cells[-1] == "":
                cells.pop()
            run.append(cells)
        else:
            flush_run()
            buffer.append(line)
    flush_run()
    if buffer:
        parts.append(("text", "\n".join(buffer)))
    return parts
