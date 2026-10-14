import argparse
import logging
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

from extraction.normalizer import (
    MIN_IMAGE_SIDE_PT,
    PageTextBuilder,
    decorative_hashes,
    file_id_for,
    image_hash,
    save_chunks,
)
from extraction.validation import DocumentValidationError, validate_document
from models.schemas import ExtractionInput
from ocr.ocr_engine import LOW_CONF_THRESHOLD, OCR_WORKERS, OcrEngine, OcrWord

logger = logging.getLogger("PdfExtractor")

__all__ = ["extract_pdf", "render_overlays", "save_chunks"]

DUPLICATE_OVERLAP = 0.5
_METADATA_FIELDS = ("author", "title", "subject", "keywords", "creator", "producer")


def _overlap_fraction(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    inter = pymupdf.Rect(a) & b
    if inter.is_empty or a.is_empty:
        return 0.0
    return inter.get_area() / a.get_area()


class _ImagePlacement:
    def __init__(self, page: pymupdf.Page, rect: pymupdf.Rect, img_size: tuple[int, int]):
        self.derotate = page.derotation_matrix
        self.clip = rect * page.rotation_matrix
        self.sx = self.clip.width / img_size[0]
        self.sy = self.clip.height / img_size[1]

    def to_page(self, x0: float, y0: float, x1: float, y1: float) -> pymupdf.Rect:
        shown = pymupdf.Rect(
            self.clip.x0 + x0 * self.sx, self.clip.y0 + y0 * self.sy,
            self.clip.x0 + x1 * self.sx, self.clip.y0 + y1 * self.sy,
        )
        return (shown * self.derotate).normalize()

    def word_rect(self, word: OcrWord) -> pymupdf.Rect:
        return self.to_page(word.left, word.top, word.left + word.width, word.top + word.height)

    def box_xywh(self, box) -> tuple[float, float, float, float]:
        r = self.to_page(*box)
        return r.x0, r.y0, r.width, r.height


def _rect_for(rects: dict[int, pymupdf.Rect], word: OcrWord) -> tuple[float, float, float, float]:
    r = rects[id(word)]
    return r.x0, r.y0, r.width, r.height


def _render_region(page: pymupdf.Page, rect: pymupdf.Rect, dpi: int) -> Image.Image:
    pix = page.get_pixmap(clip=rect * page.rotation_matrix, dpi=dpi, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _image_digest(doc: pymupdf.Document, xref: int) -> str | None:
    if xref <= 0:
        return None
    try:
        return image_hash(doc.extract_image(xref)["image"])
    except (RuntimeError, ValueError, KeyError, TypeError):
        return None


def _add_native_text(page: pymupdf.Page, builder: PageTextBuilder) -> list:
    native_rects = []
    current_line = None
    current_block = None
    for x0, y0, x1, y1, text, block_no, line_no, _ in page.get_text("words"):
        if current_block is not None and block_no != current_block:
            builder.end_block()
        elif (block_no, line_no) != current_line:
            builder.end_line()
        current_block, current_line = block_no, (block_no, line_no)
        rect = pymupdf.Rect(x0, y0, x1, y1)
        native_rects.append((text.lower(), rect))
        builder.add_word(text, rect.x0, rect.y0, rect.width, rect.height)
    builder.end_block()
    return native_rects


class _ImageJob:
    def __init__(self, rect, xref, placement, future):
        self.rect, self.xref, self.placement, self.future = rect, xref, placement, future


def _queue_images(page, builder, ocr_engine, dpi, pool) -> list[_ImageJob]:
    page_area = page.rect * page.derotation_matrix
    placements = []
    for info in page.get_image_info(xrefs=True):
        rect = pymupdf.Rect(info["bbox"]) & page_area
        if not rect.is_empty:
            placements.append((rect, info.get("xref", 0)))
    placements.sort(key=lambda p: (round(p[0].y0, 1), p[0].x0))

    ocr_available = ocr_engine is not None and ocr_engine.is_available()
    decorative = decorative_hashes()
    jobs = []
    for rect, xref in placements:
        if rect.width < MIN_IMAGE_SIDE_PT or rect.height < MIN_IMAGE_SIDE_PT:
            builder.images_skipped += 1
            logger.info("Page %d: skipped small image %.0fx%.0f pt",
                        builder.page_number, rect.width, rect.height)
            continue
        if decorative and _image_digest(page.parent, xref) in decorative:
            builder.images_decorative += 1
            continue
        if not ocr_available:
            builder.images_skipped += 1
            continue

        img = _render_region(page, rect, dpi)
        placement = _ImagePlacement(page, rect, img.size)
        future = pool.submit(ocr_engine.ocr_image_detailed, img, px_per_inch=dpi)
        jobs.append(_ImageJob(rect, xref, placement, future))

    if placements and not ocr_available and builder.images_skipped:
        logger.warning("Page %d: Tesseract unavailable, %d image(s) not OCR'd",
                       builder.page_number, builder.images_skipped)
    return jobs


def _apply_ocr(builder: PageTextBuilder, job: _ImageJob, native_rects):
    rect = job.rect
    try:
        result = job.future.result()
    except RuntimeError as exc:
        builder.images_skipped += 1
        builder.errors.append(f"OCR failed for image xref {job.xref}: {exc}")
        builder.add_unread(rect.x0, rect.y0, rect.width, rect.height, reason="image_error")
        return
    builder.images_ocr += 1

    kept = []
    for word in result.words:
        r = job.placement.word_rect(word)
        if not any(
            text == word.text.lower() and _overlap_fraction(r, nr) > DUPLICATE_OVERLAP
            for text, nr in native_rects
        ):
            kept.append((word, r))
    builder.add_ocr_words(
        [w for w, _ in kept], partial(_rect_for, {id(w): r for w, r in kept})
    )
    for region in result.unread_regions:
        builder.add_unread(*job.placement.box_xywh(region))


def _add_page_hidden_content(page: pymupdf.Page, builder: PageTextBuilder):
    for annot in page.annots() or []:
        if annot.type[1] == "Popup":
            continue
        info = annot.info
        text = " ".join(v for v in (info.get("title"), info.get("subject"), info.get("content")) if v)
        builder.add_hidden("annotation", text, f"{annot.type[1]} annotation")
    for widget in page.widgets() or []:
        value = widget.field_value
        if isinstance(value, str) and value not in ("", "Off"):
            builder.add_hidden("form_field", value, widget.field_name)
    for link in page.get_links():
        if link.get("uri"):
            builder.add_hidden("link", link["uri"])


def _document_hidden_content(doc: pymupdf.Document) -> tuple[list, dict[int, list[str]]]:
    items = []
    meta = doc.metadata or {}
    values = set()
    for field in _METADATA_FIELDS:
        value = (meta.get(field) or "").strip()
        if value:
            items.append(("metadata", value, field))
            values.add(value)
    xmp = doc.get_xml_metadata() or ""
    xmp_text = [t.strip() for t in re.findall(r">([^<>]+)<", xmp) if t.strip()]
    extra = [t for t in dict.fromkeys(xmp_text) if t not in values]
    if extra:
        items.append(("xmp_metadata", " ".join(extra), "xmp"))
    for name in doc.embfile_names():
        info = doc.embfile_info(name)
        text = " ".join(v for v in (info.get("filename"), info.get("desc")) if v)
        items.append(("embedded_file", text or name, name))

    bookmarks: dict[int, list[str]] = {}
    for _, title, target in doc.get_toc(simple=True):
        bookmarks.setdefault(max(target, 1), []).append(title)
    return items, bookmarks


class _PendingPage:
    def __init__(self, builder, native_rects, jobs):
        self.builder, self.native_rects, self.jobs = builder, native_rects, jobs

    def finish(self) -> PageTextBuilder:
        for job in self.jobs:
            _apply_ocr(self.builder, job, self.native_rects)
        return self.builder


def _start_page(page, page_number, ocr_engine, dpi, pool) -> _PendingPage:
    low_threshold = ocr_engine.low_conf_threshold if ocr_engine else 0
    builder = PageTextBuilder(page_number, low_threshold)
    native_rects, jobs = [], []
    try:
        native_rects = _add_native_text(page, builder)
        jobs = _queue_images(page, builder, ocr_engine, dpi, pool)
        _add_page_hidden_content(page, builder)
    except (RuntimeError, ValueError) as exc:
        logger.warning("Page %d could not be fully extracted: %s", page_number, exc)
        builder.errors.append(f"Page extraction failed: {exc}")
        area = page.rect * page.derotation_matrix
        builder.add_unread(area.x0, area.y0, area.width, area.height, reason="page_error")
    return _PendingPage(builder, native_rects, jobs)


def extract_pdf(
    path: str | Path,
    *,
    file_id: str | None = None,
    source_url: str | None = None,
    user_id: str | None = None,
    ocr_engine: OcrEngine | None = None,
    dpi: int = 300,
    workers: int = OCR_WORKERS,
) -> list[ExtractionInput]:
    path = Path(path)
    if validate_document(path) != "pdf":
        raise DocumentValidationError("unsupported_type", f"{path.name} is not a PDF")
    file_id = file_id or file_id_for(path)
    source_url = source_url or str(path)
    if ocr_engine is None:
        ocr_engine = OcrEngine()
    if not ocr_engine.is_available():
        logger.warning("Tesseract not found; images will not be OCR'd")

    def to_chunk(builder: PageTextBuilder) -> ExtractionInput:
        if builder.page_number == 1:
            for kind, text, location in doc_items:
                builder.add_hidden(kind, text, location)
        for title in bookmarks.get(builder.page_number, []):
            builder.add_hidden("bookmark", title)
        return ExtractionInput(
            file_id=file_id,
            file_name=path.name,
            source_url=source_url,
            user_id=user_id,
            page_number=builder.page_number,
            **builder.page_data(),
        )

    chunks = []
    pending: deque[_PendingPage] = deque()
    with pymupdf.open(path) as doc, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        doc_items, bookmarks = _document_hidden_content(doc)
        for index, page in enumerate(doc):
            pending.append(_start_page(page, index + 1, ocr_engine, dpi, pool))
            while len(pending) > max(1, workers):
                chunks.append(to_chunk(pending.popleft().finish()))
        while pending:
            chunks.append(to_chunk(pending.popleft().finish()))
    return chunks


OVERLAY_COLORS = {
    "native": (37, 99, 235),
    "ocr": (22, 163, 74),
    "low": (220, 38, 38),
    "unread": (234, 88, 12),
}


def _draw_box(draw, to_image, box, color, width=2):
    r = (pymupdf.Rect(box.x, box.y, box.x + box.w, box.y + box.h) * to_image).normalize()
    draw.rectangle([r.x0, r.y0, r.x1, r.y1], outline=color, width=width)


def render_overlays(
    path: str | Path,
    chunks: list[ExtractionInput],
    out_dir: str | Path,
    *,
    pages: set[int] | None = None,
    dpi: int = 120,
    low_conf_threshold: float = LOW_CONF_THRESHOLD,
) -> list[Path]:
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

            to_image = page.rotation_matrix * pymupdf.Matrix(scale, scale)
            for box in chunk.bounding_boxes or []:
                if box.source == "ocr" and (box.confidence or 0) < low_conf_threshold:
                    color = OVERLAY_COLORS["low"]
                else:
                    color = OVERLAY_COLORS.get(box.source, OVERLAY_COLORS["native"])
                _draw_box(draw, to_image, box, color)
            for region in chunk.unread_regions:
                _draw_box(draw, to_image, region, OVERLAY_COLORS["unread"], width=4)

            target = out_dir / f"page_{chunk.page_number:02d}.png"
            img.save(target)
            written.append(target)
    return written


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
             "(blue=native, green=OCR, red=low-confidence OCR, orange=unread ink; images contain PII)",
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
        out = save_chunks(chunks, args.save_chunks)
        print(f"Saved {len(chunks)} page chunk(s) to {out.resolve()}")


if __name__ == "__main__":
    main()
