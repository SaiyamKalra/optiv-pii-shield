# extraction/__main__.py
"""python -m extraction <file.pdf|.docx|.pptx> [--show-text] [--show-hidden] [--pages 1,2]
                       [--save-chunks FILE] [--list-images]"""

import argparse
import logging
import sys
import zipfile
from pathlib import Path

import pymupdf

from extraction.normalizer import (
    decorative_hashes,
    extract_document,
    image_hash,
    save_chunks,
)
from extraction.validation import DocumentValidationError, validate_document


def list_images(path: Path):
    """Print every embedded image's SHA-256, for adding logos to decorative_images.json."""
    kind = validate_document(path)
    known = decorative_hashes()
    rows = []
    if kind == "pdf":
        with pymupdf.open(path) as doc:
            seen = set()
            for page in doc:
                for xref, *_ in page.get_images(full=True):
                    if xref in seen:
                        continue
                    seen.add(xref)
                    info = doc.extract_image(xref)
                    rows.append((image_hash(info["image"]), f"xref {xref} on page {page.number + 1}",
                                 f"{info['width']}x{info['height']}"))
    else:
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if "/media/" in name:
                    rows.append((image_hash(zf.read(name)), name, f"{zf.getinfo(name).file_size} bytes"))
    for digest, where, size in rows:
        flag = "  [decorative]" if digest in known else ""
        print(f"{digest}  {where}  {size}{flag}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract a PDF, DOCX or PPTX page by page (slide by slide), with OCR."
    )
    parser.add_argument("file", help="Path to a .pdf, .docx or .pptx file")
    parser.add_argument(
        "--show-text", action="store_true",
        help="Print each page's extracted text (may contain PII)",
    )
    parser.add_argument(
        "--show-hidden", action="store_true",
        help="Print hidden content (metadata, comments, ...) and unread regions (may contain PII)",
    )
    parser.add_argument("--pages", help="Comma-separated page/slide numbers to show, e.g. 1,4")
    parser.add_argument(
        "--save-chunks", metavar="FILE",
        help="Save all pages as ExtractionInput JSON for the PII pipeline "
             "(python -m pii.pipeline FILE); contains PII",
    )
    parser.add_argument(
        "--list-images", action="store_true",
        help="List embedded images with their SHA-256 (for decorative_images.json) and exit",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    try:
        if args.list_images:
            list_images(Path(args.file))
            return
        chunks = extract_document(args.file)
    except DocumentValidationError as exc:
        print(f"Cannot extract: {exc} [{exc.reason}]", file=sys.stderr)
        sys.exit(2)

    selected = {int(p) for p in args.pages.split(",")} if args.pages else None

    print(f"{Path(args.file).name}: {len(chunks)} pages")
    print(f"{'page':>4}  {'method':<7} {'chars':>6} {'boxes':>6} {'mean_conf':>9} "
          f"{'low':>4} {'imgs':>4} {'skip':>4} {'deco':>4} {'unread':>6} {'hidden':>6} {'errors':>6}")
    for c in chunks:
        if selected and c.page_number not in selected:
            continue
        conf = f"{c.ocr_mean_confidence:.1f}" if c.ocr_mean_confidence is not None else "-"
        print(
            f"{c.page_number:>4}  {c.extraction_method:<7} {len(c.raw_text):>6} "
            f"{len(c.bounding_boxes or []):>6} {conf:>9} {c.low_confidence_word_count:>4} "
            f"{c.images_ocr_count:>4} {c.images_skipped_count:>4} {c.images_decorative_count:>4} "
            f"{len(c.unread_regions):>6} {len(c.hidden_content):>6} {len(c.extraction_errors):>6}"
        )
        if args.show_text:
            print(c.raw_text)
            print("-" * 80)
        if args.show_hidden:
            for item in c.hidden_content:
                print(f"    hidden {item.kind} ({item.location}): {item.text}")
            for r in c.unread_regions:
                where = f" in {r.image_ref}" if r.image_ref else ""
                print(f"    unread {r.reason}: x={r.x:.0f} y={r.y:.0f} w={r.w:.0f} h={r.h:.0f} "
                      f"[{r.coord_space}]{where}")
            for err in c.extraction_errors:
                print(f"    error: {err}")

    if args.save_chunks:
        out = save_chunks(chunks, args.save_chunks)
        print(f"Saved {len(chunks)} page chunk(s) to {out.resolve()}")


if __name__ == "__main__":
    main()
