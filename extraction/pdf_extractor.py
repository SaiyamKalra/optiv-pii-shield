# extraction/pdf_extractor.py

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from models.schemas import BoundingBox, ExtractionInput
from ocr.ocr_engine import LOW_CONF_THRESHOLD, OcrEngine, OcrWord

logger = logging.getLogger("PdfExtractor")

# Images smaller than this on either side (PDF points) are logos/icons and are not OCR'd.
MIN_IMAGE_SIDE_PT = 40.0
# An OCR word overlapping a native word by more than this fraction is a duplicate.
DUPLICATE_OVERLAP = 0.5


class _PageTextBuilder:
    """Builds raw_text and word boxes together so char offsets always line up."""

    def __init__(self, page_number: int):
        self.page_number = page_number
        self.parts: list[str] = []
        self.length = 0
        self.boxes: list[BoundingBox] = []
        self.line_open = False

    def _append(self, s: str):
        self.parts.append(s)
        self.length += len(s)

    def add_word(self, text, rect: pymupdf.Rect, source="native", confidence=None):
        if self.line_open:
            self._append(" ")
        start = self.length
        self._append(text)
        self.line_open = True
        self.boxes.append(
            BoundingBox(
                page=self.page_number,
                text=text,
                x=rect.x0,
                y=rect.y0,
                w=rect.width,
                h=rect.height,
                char_start=start,
                char_end=self.length,
                source=source,
                confidence=confidence,
            )
        )

    def end_line(self):
        if self.line_open:
            self._append("\n")
            self.line_open = False

    def end_block(self):
        self.end_line()
        if self.parts and not "".join(self.parts[-2:]).endswith("\n\n"):
            self._append("\n")

    @property
    def text(self) -> str:
        return "".join(self.parts).rstrip("\n")


def _file_id_for(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"doc_{digest[:12]}"


def _overlap_fraction(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    inter = pymupdf.Rect(a) & b
    if inter.is_empty or a.is_empty:
        return 0.0
    return inter.get_area() / a.get_area()


def _render_region(page: pymupdf.Page, rect: pymupdf.Rect, dpi: int) -> Image.Image:
    pix = page.get_pixmap(clip=rect, dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _ocr_words_to_page(
    words: list[OcrWord], region: pymupdf.Rect, img: Image.Image
) -> list[tuple[OcrWord, pymupdf.Rect]]:
    sx = region.width / img.width
    sy = region.height / img.height
    placed = []
    for w in words:
        x0 = region.x0 + w.left * sx
        y0 = region.y0 + w.top * sy
        placed.append((w, pymupdf.Rect(x0, y0, x0 + w.width * sx, y0 + w.height * sy)))
    return placed


def _extract_page(
    page: pymupdf.Page, page_number: int, ocr_engine: OcrEngine | None, dpi: int
) -> dict:
    builder = _PageTextBuilder(page_number)

    # 1. Native text layer, grouped into lines in PyMuPDF's reading order.
    native_words = page.get_text("words")
    native_rects = []
    current_line = None
    current_block = None
    for x0, y0, x1, y1, text, block_no, line_no, _ in native_words:
        if current_block is not None and block_no != current_block:
            builder.end_block()
        elif (block_no, line_no) != current_line:
            builder.end_line()
        current_block, current_line = block_no, (block_no, line_no)
        rect = pymupdf.Rect(x0, y0, x1, y1)
        native_rects.append((text.lower(), rect))
        builder.add_word(text, rect)
    builder.end_block()

    # 2. Every embedded image placement on the page, top-to-bottom, left-to-right.
    placements = []
    for info in page.get_image_info(xrefs=True):
        rect = pymupdf.Rect(info["bbox"]) & page.rect
        if rect.is_empty:
            continue
        placements.append(rect)
    placements.sort(key=lambda r: (round(r.y0, 1), r.x0))

    images_ocr = 0
    images_skipped = 0
    ocr_confidences: list[float] = []
    ocr_available = ocr_engine is not None and ocr_engine.is_available()

    for rect in placements:
        if rect.width < MIN_IMAGE_SIDE_PT or rect.height < MIN_IMAGE_SIDE_PT:
            images_skipped += 1
            logger.info(
                "Page %d: skipped small image %.0fx%.0f pt",
                page_number, rect.width, rect.height,
            )
            continue
        if not ocr_available:
            images_skipped += 1
            continue

        img = _render_region(page, rect, dpi)
        placed = _ocr_words_to_page(ocr_engine.ocr_image(img), rect, img)
        images_ocr += 1

        # 3. Drop OCR words that duplicate native text already captured.
        kept = [
            (w, r) for w, r in placed
            if not any(
                text == w.text.lower() and _overlap_fraction(r, nr) > DUPLICATE_OVERLAP
                for text, nr in native_rects
            )
        ]

        # 4. Append in Tesseract block/paragraph/line order.
        current_block = None
        current_line = None
        for w, r in kept:
            if current_block is not None and w.block != current_block:
                builder.end_block()
            elif (w.block, w.par, w.line) != current_line:
                builder.end_line()
            current_block, current_line = w.block, (w.block, w.par, w.line)
            builder.add_word(w.text, r, source="ocr", confidence=w.conf)
            ocr_confidences.append(w.conf)
        builder.end_block()

    if placements and not ocr_available and images_skipped:
        logger.warning(
            "Page %d: Tesseract unavailable, %d image(s) not OCR'd",
            page_number, images_skipped,
        )

    has_native = bool(native_words)
    has_ocr = bool(ocr_confidences)
    if has_ocr and has_native:
        method = "hybrid"
    elif has_ocr or (images_ocr and not has_native):
        method = "ocr"
    else:
        method = "native"

    low_threshold = ocr_engine.low_conf_threshold if ocr_engine else 0
    return {
        "raw_text": builder.text,
        "bounding_boxes": builder.boxes,
        "ocr_used": images_ocr > 0,
        "extraction_method": method,
        "ocr_mean_confidence": (
            sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else None
        ),
        "low_confidence_word_count": sum(c < low_threshold for c in ocr_confidences),
        "images_ocr_count": images_ocr,
        "images_skipped_count": images_skipped,
    }


def extract_pdf(
    path: str | Path,
    *,
    file_id: str | None = None,
    source_url: str | None = None,
    user_id: str | None = None,
    ocr_engine: OcrEngine | None = None,
    dpi: int = 300,
) -> list[ExtractionInput]:
    """Extract a PDF into one ExtractionInput per page, OCR'ing every embedded image."""
    path = Path(path)
    file_id = file_id or _file_id_for(path)
    source_url = source_url or str(path)
    if ocr_engine is None:
        ocr_engine = OcrEngine()
    if not ocr_engine.is_available():
        logger.warning("Tesseract not found; images will not be OCR'd")

    chunks = []
    with pymupdf.open(path) as doc:
        for index, page in enumerate(doc):
            page_number = index + 1
            page_data = _extract_page(page, page_number, ocr_engine, dpi)
            chunks.append(
                ExtractionInput(
                    file_id=file_id,
                    file_name=path.name,
                    source_url=source_url,
                    user_id=user_id,
                    page_number=page_number,
                    **page_data,
                )
            )
    return chunks


OVERLAY_COLORS = {
    "native": (37, 99, 235),   # blue: real text layer
    "ocr": (22, 163, 74),      # green: OCR word at or above the confidence threshold
    "low": (220, 38, 38),      # red: OCR word below the threshold
}


def render_overlays(
    path: str | Path,
    chunks: list[ExtractionInput],
    out_dir: str | Path,
    *,
    pages: set[int] | None = None,
    dpi: int = 120,
    low_conf_threshold: float = LOW_CONF_THRESHOLD,
) -> list[Path]:
    """Save a PNG per page with every extracted word box drawn on it.

    The images show the original page, so they contain whatever PII the page does.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72
    written = []

    with pymupdf.open(path) as doc:
        for chunk in chunks:
            if pages and chunk.page_number not in pages:
                continue
            page = doc[chunk.page_number - 1]
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            draw = ImageDraw.Draw(img)

            for box in chunk.bounding_boxes or []:
                if box.source == "ocr" and (box.confidence or 0) < low_conf_threshold:
                    color = OVERLAY_COLORS["low"]
                else:
                    color = OVERLAY_COLORS.get(box.source, OVERLAY_COLORS["native"])
                draw.rectangle(
                    [box.x * scale, box.y * scale,
                     (box.x + box.w) * scale, (box.y + box.h) * scale],
                    outline=color,
                    width=2,
                )

            target = out_dir / f"page_{chunk.page_number:02d}.png"
            img.save(target)
            written.append(target)
    return written


def save_chunks(chunks: list[ExtractionInput], path: str | Path) -> Path:
    """Write extracted pages to JSON so later stages can load them without re-running OCR."""
    path = Path(path)
    data = [c.model_dump(mode="json") for c in chunks]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Extract a PDF page by page with OCR.")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--show-text", action="store_true",
        help="Print each page's extracted text (may contain PII)",
    )
    parser.add_argument("--pages", help="Comma-separated page numbers to show, e.g. 7,18")
    parser.add_argument(
        "--overlay", metavar="DIR",
        help="Save page images with word boxes drawn on them to DIR "
             "(blue=native, green=OCR, red=low-confidence OCR; images contain PII)",
    )
    parser.add_argument(
        "--save-chunks", metavar="FILE",
        help="Save all pages as ExtractionInput JSON for the PII pipeline "
             "(python -m pii.pipeline FILE); contains PII",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    chunks = extract_pdf(args.pdf)
    selected = {int(p) for p in args.pages.split(",")} if args.pages else None

    print(f"{chunks[0].file_name if chunks else args.pdf}: {len(chunks)} pages")
    print(f"{'page':>4}  {'method':<7} {'words':>6} {'mean_conf':>9} {'low':>4} {'imgs':>4} {'skip':>4}")
    for c in chunks:
        if selected and c.page_number not in selected:
            continue
        conf = f"{c.ocr_mean_confidence:.1f}" if c.ocr_mean_confidence is not None else "-"
        print(
            f"{c.page_number:>4}  {c.extraction_method:<7} {len(c.bounding_boxes or []):>6} "
            f"{conf:>9} {c.low_confidence_word_count:>4} "
            f"{c.images_ocr_count:>4} {c.images_skipped_count:>4}"
        )
        if args.show_text:
            print(c.raw_text)
            print("-" * 80)

    if args.overlay:
        written = render_overlays(args.pdf, chunks, args.overlay, pages=selected)
        print(f"Saved {len(written)} overlay image(s) to {Path(args.overlay).resolve()}")

    if args.save_chunks:
        # Always save every page: the pipeline needs the whole document for consistent tokens.
        out = save_chunks(chunks, args.save_chunks)
        print(f"Saved {len(chunks)} page chunk(s) to {out.resolve()}")


if __name__ == "__main__":
    main()
