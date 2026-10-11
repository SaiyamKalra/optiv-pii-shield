"""Render sanitized pages as LLM-ready Markdown.

Tables are rebuilt in two ways:

* From word geometry (PDF). When the extractor's word boxes are supplied,
  words are grouped into rows and columns by their x/y position, so a table
  in the PDF becomes a real Markdown table.
* From text (DOCX/PPTX, or any text whose cells are separated by tabs, 2+
  spaces, or pipes). Consecutive lines with the same number of cells become
  a table.

Everything written to the output is built from sanitized tokens. A PII value
never appears in the Markdown, including inside tables.
"""

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import median
from typing import Any

from models.schemas import BoundingBox, ExtractionInput, PIIOutput

# Rough estimate for English text. Real tokenizers average about 4 chars/token.
CHARS_PER_TOKEN = 4.0

# Token shape produced by the PII pipeline, e.g. [PERSON_001], [EMAIL_ADDRESS_002].
_TOKEN_RE = re.compile(r"\[[A-Z][A-Z_]*_\d+\]")
_PAGE_OF_RE = re.compile(r"(?i)page\s+\d+\s+(?:of|/)\s+\d+")
_CONFIDENTIAL_RE = re.compile(r"(?i)confidential\s*(?:-|\|)?\s*internal\s+use\s+only")
_RIGHTS_RE = re.compile(r"(?i)all\s+rights\s+reserved\.?")
_HR_RE = re.compile(r"[-=*_ ]{3,}")

# Values that YAML would read as something other than a plain string.
_PLAIN_YAML_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\- ()/]*")
_YAML_SPECIAL_RE = re.compile(r"(?i)true|false|yes|no|on|off|null|~|[-+]?\d[\d.]*")

# Geometry tuning. Multiples of the median word height unless noted.
ROW_JOIN_FACTOR = 0.5     # a word joins a row if its centre is this close to the row's
CELL_GAP_MIN_PT = 8.0     # horizontal gap (points) that separates two cells in a row
CELL_GAP_FACTOR = 1.0
MERGE_GAP_PT = 2.0        # column extents closer than this (points) are one column
TABLE_ROW_GAP = 1.6       # max vertical gap between two rows of the same table
CONTINUATION_GAP = 0.6    # max gap for a wrapped line that continues a table cell
PARAGRAPH_GAP = 0.8       # vertical gap that starts a new paragraph
MIN_TABLE_ROWS = 2


@dataclass(frozen=True)
class _Cell:
    x0: float
    x1: float
    text: str


@dataclass(frozen=True)
class _Row:
    y0: float
    y1: float
    cells: tuple[_Cell, ...]


class _Blocks:
    """Collects paragraphs and tables in reading order."""

    def __init__(self):
        self.items: list[tuple[str, Any]] = []
        self._para: list[str] = []

    def line(self, text: str):
        self._para.append(text)

    def break_para(self):
        if self._para:
            self.items.append(("para", self._para))
            self._para = []

    def table(self, rows: list[list[str]]):
        self.break_para()
        self.items.append(("table", rows))

    def finish(self) -> list[tuple[str, Any]]:
        self.break_para()
        return self.items


def _yaml_scalar(value: Any) -> str:
    """Write a value as a YAML scalar, quoting it when a plain form would break."""
    text = str(value)
    if _PLAIN_YAML_RE.fullmatch(text) and not _YAML_SPECIAL_RE.fullmatch(text):
        return text
    return json.dumps(text, ensure_ascii=False)


def _cell_text(value: str) -> str:
    """Make a value safe inside one Markdown table cell."""
    return re.sub(r"\s+", " ", value).strip().replace("|", "\\|")


class TokenOptimizer:
    def __init__(
        self,
        strip_headers_footers: bool = True,
        normalize_whitespace: bool = True,
        include_page_markers: bool = True,
        include_metadata_header: bool = True,
        render_tables: bool = True,
    ):
        self.strip_headers_footers = strip_headers_footers
        self.normalize_whitespace = normalize_whitespace
        self.include_page_markers = include_page_markers
        self.include_metadata_header = include_metadata_header
        self.render_tables = render_tables

    def optimize_output(
        self,
        pii_outputs: list[PIIOutput],
        raw_original_text: str = "",
        chunks: list[ExtractionInput] | None = None,
    ) -> str:
        """Build the Markdown document.

        *chunks* are the extractor's pages (same page numbers as *pii_outputs*).
        When given, PDF tables are rebuilt from word positions. Without them,
        tables are found in the text itself.
        """
        if not pii_outputs:
            return ""

        chunk_by_page = {c.page_number: c for c in (chunks or [])}
        pages = []
        for pii in pii_outputs:
            body = self._render_page(pii, chunk_by_page.get(pii.page_number))
            if not body:
                continue
            if self.include_page_markers:
                pages.append(f"<!-- Page {pii.page_number} -->\n{body}")
            else:
                pages.append(body)

        body_markdown = "\n\n---\n\n".join(pages)

        if self.include_metadata_header:
            first_doc = pii_outputs[0]
            metrics = self.calculate_metrics(raw_original_text, body_markdown)
            header = (
                "---\n"
                f"title: {_yaml_scalar(first_doc.file_name)}\n"
                f"file_id: {_yaml_scalar(first_doc.file_id)}\n"
                f"processed_at: {datetime.now(UTC).isoformat()}\n"
                f"total_pages: {len(pii_outputs)}\n"
                f"estimated_tokens_saved: {metrics['tokens_saved']} "
                f"({metrics['savings_percentage']}%)\n"
                "---\n\n"
            )
            return header + body_markdown
        return body_markdown

    @staticmethod
    def calculate_metrics(raw_text: str, optimized_text: str) -> dict[str, Any]:
        raw_chars = len(raw_text or "")
        opt_chars = len(optimized_text or "")
        est_raw_tokens = math.ceil(raw_chars / CHARS_PER_TOKEN)
        est_opt_tokens = math.ceil(opt_chars / CHARS_PER_TOKEN)
        tokens_saved = max(0, est_raw_tokens - est_opt_tokens)
        savings_pct = (
            round(tokens_saved / est_raw_tokens * 100, 2) if est_raw_tokens else 0.0
        )
        return {
            "raw_characters": raw_chars,
            "optimized_characters": opt_chars,
            "estimated_raw_tokens": est_raw_tokens,
            "estimated_optimized_tokens": est_opt_tokens,
            "tokens_saved": tokens_saved,
            "savings_percentage": savings_pct,
        }

    # ── page rendering ───────────────────────────────────────────────────────

    def _render_page(self, pii: PIIOutput, chunk: ExtractionInput | None) -> str:
        if not (pii.sanitized_text or "").strip():
            return ""
        blocks = None
        if chunk is not None and self.render_tables:
            blocks = self._geometry_blocks(pii, chunk)
        if blocks is None:
            blocks = self._text_blocks(pii.sanitized_text or "")
        return self._render_blocks(blocks)

    def _render_blocks(self, blocks: list[tuple[str, Any]]) -> str:
        parts = []
        for kind, payload in blocks:
            if kind == "para":
                # Lines inside one paragraph are hard breaks, so Markdown keeps them apart.
                parts.append("  \n".join(payload))
            else:
                parts.append(self._markdown_table(payload))
        return "\n\n".join(parts)

    @staticmethod
    def _markdown_table(rows: list[list[str]]) -> str:
        ncols = max(len(r) for r in rows)

        def fmt(cells: list[str]) -> str:
            padded = cells + [""] * (ncols - len(cells))
            return "| " + " | ".join(_cell_text(c) for c in padded) + " |"

        lines = [fmt(rows[0]), "| " + " | ".join(["---"] * ncols) + " |"]
        lines += [fmt(r) for r in rows[1:]]
        return "\n".join(lines)

    # ── text-based tables (DOCX / PPTX / plain text) ─────────────────────────

    def _text_blocks(self, text: str) -> list[tuple[str, Any]]:
        lines = text.replace("\r\n", "\n").split("\n")
        blocks = _Blocks()
        i = 0
        while i < len(lines):
            cells = self._split_text_row(lines[i])
            if cells:
                rows = [cells]
                j = i + 1
                while j < len(lines):
                    nxt = self._split_text_row(lines[j])
                    if not nxt or len(nxt) != len(cells):
                        break
                    rows.append(nxt)
                    j += 1
                if len(rows) >= MIN_TABLE_ROWS:
                    blocks.table([[self._clean_cell(c) for c in r] for r in rows])
                    i = j
                    continue

            line = self._clean_line(lines[i])
            if line:
                blocks.line(self._escape_line(line))
            else:
                blocks.break_para()
            i += 1
        return blocks.finish()

    @staticmethod
    def _split_text_row(line: str) -> list[str] | None:
        """Split one line into cells, or return None if it is not table-like."""
        if "\t" in line:
            cells = [c.strip() for c in line.split("\t")]
            return cells if len(cells) >= 2 and any(cells) else None
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3:
            return [c.strip() for c in stripped[1:-1].split("|")]
        # Space-separated columns are easy to confuse with prose that has a
        # double space after a sentence. Accept them only when every part is
        # short and none ends like a sentence.
        parts = [p.strip() for p in re.split(r"\s{2,}", stripped) if p.strip()]
        if len(parts) < 2:
            return None
        if any(len(p) > 40 or p[-1] in ".!?" for p in parts):
            return None
        return parts

    # ── geometry-based tables (PDF) ──────────────────────────────────────────

    def _geometry_blocks(
        self, pii: PIIOutput, chunk: ExtractionInput
    ) -> list[tuple[str, Any]] | None:
        """Rebuild the page from word boxes. Returns None to fall back to text."""
        raw = chunk.raw_text or ""
        if not self._entities_match(raw, pii):
            return None
        boxes = [
            b for b in (chunk.bounding_boxes or [])
            if b.char_start is not None and b.char_end is not None
        ]
        if not boxes:
            return None

        entities = [
            (d.start, d.end, d.token)
            for d in sorted(pii.detected_entities, key=lambda d: d.start)
        ]
        emitted: set[int] = set()
        words: list[tuple[BoundingBox, str]] = []
        for box in sorted(boxes, key=lambda b: b.char_start):
            if not 0 <= box.char_start < box.char_end <= len(raw):
                return None
            word = self._sanitize_range(raw, box.char_start, box.char_end, entities, emitted)
            if word:
                words.append((box, word))
        if not words:
            return None

        heights = [box.h for box, _ in words if box.h > 0]
        med_h = median(heights) if heights else 10.0
        rows = self._group_rows(words, med_h)
        blocks = self._rows_to_blocks(rows, med_h)

        # Only use the geometry result if it keeps every token exactly once.
        if not self._is_faithful(self._render_blocks(blocks), pii):
            return None
        return blocks

    @staticmethod
    def _is_faithful(rendered: str, pii: PIIOutput) -> bool:
        """Every token must appear as often as in the sanitized text, and no
        original PII value may appear anywhere in the output."""
        expected = Counter(_TOKEN_RE.findall(pii.sanitized_text or ""))
        if Counter(_TOKEN_RE.findall(rendered)) != expected:
            return False
        return not any(
            d.original_value and d.original_value in rendered
            for d in pii.detected_entities
        )

    @staticmethod
    def _entities_match(raw: str, pii: PIIOutput) -> bool:
        """Entity offsets must point at the same text in the page's raw_text."""
        return all(
            0 <= d.start < d.end <= len(raw) and raw[d.start:d.end] == d.original_value
            for d in pii.detected_entities
        )

    @staticmethod
    def _sanitize_range(
        raw: str,
        start: int,
        end: int,
        entities: list[tuple[int, int, str]],
        emitted: set[int],
    ) -> str:
        """Text of raw[start:end] with each entity replaced by its token, once."""
        out: list[str] = []
        pos = start
        while pos < end:
            cover = None
            for idx, (s, e, token) in enumerate(entities):
                if s <= pos < e:
                    cover = (idx, e, token)
                    break
            if cover:
                idx, e, token = cover
                if idx not in emitted:
                    out.append(token)
                    emitted.add(idx)
                pos = min(e, end)
            else:
                nxt = end
                for s, _, _ in entities:
                    if pos < s < nxt:
                        nxt = s
                out.append(raw[pos:nxt])
                pos = nxt
        return "".join(out).strip()

    @staticmethod
    def _group_rows(words: list[tuple[BoundingBox, str]], med_h: float) -> list[_Row]:
        ordered = sorted(words, key=lambda bw: (bw[0].y + bw[0].h / 2, bw[0].x))
        groups: list[list[tuple[BoundingBox, str]]] = []
        centres: list[float] = []
        for box, text in ordered:
            cy = box.y + box.h / 2
            if groups and abs(cy - centres[-1]) <= ROW_JOIN_FACTOR * med_h:
                groups[-1].append((box, text))
                centres[-1] += (cy - centres[-1]) / len(groups[-1])
            else:
                groups.append([(box, text)])
                centres.append(cy)

        gap_tol = max(CELL_GAP_MIN_PT, CELL_GAP_FACTOR * med_h)
        rows: list[_Row] = []
        for group in groups:
            group.sort(key=lambda bw: bw[0].x)
            cells: list[list] = []  # [x0, x1, [words]]
            for box, text in group:
                x1 = box.x + box.w
                if cells and box.x - cells[-1][1] <= gap_tol:
                    cells[-1][1] = max(cells[-1][1], x1)
                    cells[-1][2].append(text)
                else:
                    cells.append([box.x, x1, [text]])
            rows.append(
                _Row(
                    y0=min(b.y for b, _ in group),
                    y1=max(b.y + b.h for b, _ in group),
                    cells=tuple(
                        _Cell(x0, x1, " ".join(ws)) for x0, x1, ws in cells
                    ),
                )
            )
        return rows

    def _rows_to_blocks(self, rows: list[_Row], h: float) -> list[tuple[str, Any]]:
        blocks = _Blocks()
        para_gap = PARAGRAPH_GAP * h
        table_gap = TABLE_ROW_GAP * h
        cont_gap = CONTINUATION_GAP * h
        n = len(rows)
        i = 0
        prev: _Row | None = None
        while i < n:
            row = rows[i]
            if prev is not None and row.y0 - prev.y1 > para_gap:
                blocks.break_para()

            if len(row.cells) >= 2:
                j = i + 1
                while (
                    j < n
                    and len(rows[j].cells) >= 2
                    and rows[j].y0 - rows[j - 1].y1 <= table_gap
                ):
                    j += 1
                table = self._table_from_run(rows[i:j], h)
                if table is not None:
                    grid, bounds = table
                    k = j
                    # The table keeps going while rows fit its columns. A wrapped
                    # line joins the cell above it; a full row is placed as a new row.
                    while k < n and rows[k].y0 - rows[k - 1].y1 <= table_gap:
                        row_k = rows[k]
                        if len(row_k.cells) == 1:
                            if (
                                row_k.y0 - rows[k - 1].y1 <= cont_gap
                                and self._absorb_continuation(grid, bounds, row_k.cells[0])
                            ):
                                k += 1
                                continue
                            break
                        placed = self._place_row(row_k, bounds)
                        if placed is None:
                            break
                        grid.append(placed)
                        k += 1
                    blocks.table([[self._clean_cell(t) for t in r] for r in grid])
                    prev = rows[k - 1]
                    i = k
                    continue

            text = " ".join(
                t for t in (self._clean_cell(c.text) for c in row.cells) if t
            )
            if text:
                blocks.line(self._escape_line(text))
            else:
                blocks.break_para()
            prev = row
            i += 1
        return blocks.finish()

    @staticmethod
    def _table_from_run(run: list[_Row], h: float) -> tuple[list[list[str]], list[tuple[float, float]]] | None:
        """Turn a run of multi-cell rows into a grid. Columns are the x-ranges
        that no cell in the run crosses. Returns None when the run is not a
        clean grid."""
        if len(run) < MIN_TABLE_ROWS:
            return None
        intervals = sorted((c.x0, c.x1) for row in run for c in row.cells)
        segments: list[list[float]] = []
        for x0, x1 in intervals:
            if segments and x0 <= segments[-1][1] + MERGE_GAP_PT:
                segments[-1][1] = max(segments[-1][1], x1)
            else:
                segments.append([x0, x1])
        if len(segments) < 2:
            return None

        grid = [[""] * len(segments) for _ in run]
        for ri, row in enumerate(run):
            for cell in row.cells:
                col = next(
                    (
                        c for c, (s0, s1) in enumerate(segments)
                        if s0 - MERGE_GAP_PT <= cell.x0 <= s1 + MERGE_GAP_PT
                    ),
                    None,
                )
                # Two cells of one row in one column means the grid is wrong.
                if col is None or grid[ri][col]:
                    return None
                grid[ri][col] = cell.text
        return grid, [(s0, s1) for s0, s1 in segments]

    @staticmethod
    def _place_row(row: _Row, bounds: list[tuple[float, float]]) -> list[str] | None:
        """Put one row into existing columns, or return None if it does not fit."""
        out = [""] * len(bounds)
        for cell in row.cells:
            col = next(
                (
                    c for c, (s0, s1) in enumerate(bounds)
                    if s0 - MERGE_GAP_PT <= cell.x0 <= s1 + MERGE_GAP_PT
                ),
                None,
            )
            if col is None or out[col]:
                return None
            out[col] = cell.text
        return out

    @staticmethod
    def _absorb_continuation(grid, bounds, cell: _Cell) -> bool:
        """Append a wrapped line to the last table row's cell in the same column."""
        for col, (s0, s1) in enumerate(bounds):
            if s0 - MERGE_GAP_PT <= cell.x0 and cell.x1 <= s1 + MERGE_GAP_PT:
                if grid[-1][col]:
                    grid[-1][col] += " " + cell.text
                    return True
                return False
        return False

    # ── cleaning helpers ─────────────────────────────────────────────────────

    def _strip_boilerplate(self, text: str) -> str:
        if not self.strip_headers_footers:
            return text
        for pattern in (_PAGE_OF_RE, _CONFIDENTIAL_RE, _RIGHTS_RE):
            text = pattern.sub("", text)
        return text

    def _clean_cell(self, text: str) -> str:
        return re.sub(r"\s+", " ", self._strip_boilerplate(text)).strip()

    def _clean_line(self, text: str) -> str:
        text = self._strip_boilerplate(text)
        if self.normalize_whitespace:
            text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    @staticmethod
    def _escape_line(text: str) -> str:
        """Stop body text from turning into a heading, quote or rule by accident."""
        if text.startswith(("#", ">")) or _HR_RE.fullmatch(text):
            return "\\" + text
        return text
