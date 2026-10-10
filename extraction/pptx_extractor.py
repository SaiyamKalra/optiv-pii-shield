# extraction/pptx_extractor.py

import logging
from functools import partial
from pathlib import Path
from xml.etree import ElementTree

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.exc import PackageNotFoundError

from extraction.normalizer import (
    EMU_PER_POINT,
    MIN_IMAGE_SIDE_PT,
    ImageOcrCache,
    PageTextBuilder,
    drawing_part_text,
    file_id_for,
    ooxml_hidden_content,
    record_image_outcome,
)
from extraction.validation import DocumentValidationError, validate_document
from models.schemas import ExtractionInput
from ocr.ocr_engine import OcrEngine, OcrWord

logger = logging.getLogger("PptxExtractor")

DIAGRAM_URI = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
# Errors python-pptx raises on malformed shapes; the shape is reported, not fatal.
SHAPE_ERRORS = (AttributeError, KeyError, ValueError, TypeError, ElementTree.ParseError)


def _pt(emu: int | None) -> float | None:
    return None if emu is None else emu / EMU_PER_POINT


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _reading_order(shapes) -> list:
    return sorted(shapes, key=lambda s: (s.top or 0, s.left or 0))


def _image_ref(shape) -> str:
    """The picture's part name inside the package, e.g. '/ppt/media/image1.png'."""
    try:
        rid = shape._element.blip_rId
        return str(shape.part.related_part(rid).partname)
    except (AttributeError, KeyError):
        return f"sha1:{shape.image.sha1}"


def _to_slide(shape, img_size, x0, y0, x1, y1) -> tuple[float, float, float, float]:
    """Map an image-pixel box to slide points, accounting for cropping of the picture."""
    img_w, img_h = img_size
    crop_l, crop_r = shape.crop_left or 0.0, shape.crop_right or 0.0
    crop_t, crop_b = shape.crop_top or 0.0, shape.crop_bottom or 0.0
    sx = _pt(shape.width) / (img_w * max(1.0 - crop_l - crop_r, 1e-6))
    sy = _pt(shape.height) / (img_h * max(1.0 - crop_t - crop_b, 1e-6))
    x = _pt(shape.left) + (x0 - crop_l * img_w) * sx
    y = _pt(shape.top) + (y0 - crop_t * img_h) * sy
    return x, y, (x1 - x0) * sx, (y1 - y0) * sy


def _word_on_slide(shape, img_size, word: OcrWord):
    return _to_slide(shape, img_size, word.left, word.top, word.left + word.width, word.top + word.height)


def _box_on_slide(shape, img_size, box):
    return _to_slide(shape, img_size, *box)


def _word_in_image(word: OcrWord):
    return word.left, word.top, word.width, word.height


def _box_in_image(box):
    return box[0], box[1], box[2] - box[0], box[3] - box[1]


def _add_picture(shape, builder: PageTextBuilder, cache: ImageOcrCache, in_group: bool):
    width, height = _pt(shape.width), _pt(shape.height)
    if width is not None and height is not None and (
        width < MIN_IMAGE_SIDE_PT or height < MIN_IMAGE_SIDE_PT
    ):
        builder.images_skipped += 1
        logger.info("Slide %d: skipped small picture %r", builder.page_number, shape.name)
        return

    ref = _image_ref(shape)
    outcome = cache.ocr(ref, shape.image.blob, display_width_pt=width)
    # Group members use the group's own coordinate space, so fall back to image pixels.
    on_slide = not in_group and None not in (shape.left, shape.top, width, height)
    if on_slide and outcome.size:
        place = partial(_word_on_slide, shape, outcome.size)
        place_box = partial(_box_on_slide, shape, outcome.size)
        coord_space = "page_pt"
    else:
        place, place_box, coord_space = _word_in_image, _box_in_image, "image_px"
    # If the picture can't be read at all, report its whole area on the slide.
    whole = (_pt(shape.left), _pt(shape.top), width, height) if on_slide else None
    record_image_outcome(
        builder, outcome, ref, place, place_box,
        coord_space=coord_space if outcome.status == "ok" else "page_pt",
        whole_image=whole,
    )


def _graphic_data(shape):
    return next((el for el in shape._element.iter() if _local(el.tag) == "graphicData"), None)


def _add_graphic_frame(shape, builder: PageTextBuilder):
    """Charts and SmartArt hold visible text in their own parts; OLE objects are flagged."""
    if getattr(shape, "has_chart", False) and shape.has_chart:
        builder.add_text(drawing_part_text(shape.chart_part.blob))
        builder.end_block()
        return
    data = _graphic_data(shape)
    if data is None:
        return
    uri = data.get("uri", "")
    if uri == DIAGRAM_URI:
        rel_ids = next((el for el in data.iter() if _local(el.tag) == "relIds"), None)
        rid = rel_ids.get(f"{{{R_NS}}}dm") if rel_ids is not None else None
        if rid:
            builder.add_text(drawing_part_text(shape.part.related_part(rid).blob))
            builder.end_block()
    elif "ole" in uri.lower():
        ole = next((el for el in data.iter() if _local(el.tag) == "oleObj"), None)
        name = (ole.get("progId") or ole.get("name") or "") if ole is not None else ""
        builder.add_hidden("embedded_object", name or "embedded object", shape.name)


def _add_alt_text(shape, builder: PageTextBuilder):
    nv = next((el for el in shape._element.iter() if _local(el.tag) == "cNvPr"), None)
    if nv is not None:
        text = " ".join(v for v in (nv.get("title"), nv.get("descr")) if v)
        builder.add_hidden("alt_text", text, shape.name)


def _add_shape(shape, builder: PageTextBuilder, cache: ImageOcrCache, in_group=False):
    _add_alt_text(shape, builder)
    if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
        for child in _reading_order(shape.shapes):
            _add_shape(child, builder, cache, in_group=True)
        return

    if getattr(shape, "has_table", False) and shape.has_table:
        for row in shape.table.rows:
            for i, cell in enumerate(row.cells):
                if i:
                    builder.add_text("|")
                builder.add_text(cell.text)
            builder.end_line()
        builder.end_block()
        return

    if _graphic_data(shape) is not None:  # chart, SmartArt or OLE object
        _add_graphic_frame(shape, builder)

    if getattr(shape, "has_text_frame", False) and shape.has_text_frame:
        for paragraph in shape.text_frame.paragraphs:
            builder.add_text(paragraph.text)
            builder.end_line()
        builder.end_block()

    try:
        has_image = shape.shape_type == MSO_SHAPE_TYPE.PICTURE or shape.image is not None
    except (AttributeError, ValueError):
        has_image = False
    if has_image:
        _add_picture(shape, builder, cache, in_group)


def _comment_text(blob: bytes) -> list[str]:
    """Text of each comment in a legacy or modern PowerPoint comments part."""
    try:
        root = ElementTree.fromstring(blob)
    except ElementTree.ParseError:
        return []
    comments = []
    for el in root:
        texts = [t.text for t in el.iter() if _local(t.tag) in ("text", "t") and t.text]
        if texts:
            comments.append(" ".join(texts))
    return comments


def _comment_authors(prs) -> list[str]:
    names = []
    for rel in prs.part.rels.values():
        if "author" in rel.reltype.lower():
            try:
                root = ElementTree.fromstring(rel.target_part.blob)
            except ElementTree.ParseError:
                continue
            names += [el.get("name") for el in root.iter() if el.get("name")]
    return names


def _add_slide_comments(slide, builder: PageTextBuilder):
    for rel in slide.part.rels.values():
        if "comment" in rel.reltype.lower() and not rel.is_external:
            for i, text in enumerate(_comment_text(rel.target_part.blob), start=1):
                builder.add_hidden("comment", text, f"comment {i}")


def _open(path: Path):
    try:
        return Presentation(str(path))
    except (PackageNotFoundError, KeyError, ValueError, ElementTree.ParseError) as exc:
        raise DocumentValidationError("corrupt", f"{path.name} can't be read as PPTX: {exc}") from exc


def extract_pptx(
    path: str | Path,
    *,
    file_id: str | None = None,
    source_url: str | None = None,
    user_id: str | None = None,
    ocr_engine: OcrEngine | None = None,
) -> list[ExtractionInput]:
    """Extract a PPTX into one ExtractionInput per slide, OCR'ing every picture.

    Slide text, tables, charts, SmartArt and speaker notes are read natively (no word
    boxes). OCR boxes from pictures are in slide points, or image pixels for pictures
    in groups. Comments, alt text, document properties and embedded objects go to
    hidden_content. Raises DocumentValidationError if the file isn't a readable PPTX.
    """
    path = Path(path)
    if validate_document(path) != "pptx":
        raise DocumentValidationError("unsupported_type", f"{path.name} is not a PPTX")
    file_id = file_id or file_id_for(path)
    source_url = source_url or str(path)
    ocr_engine = ocr_engine or OcrEngine()
    cache = ImageOcrCache(ocr_engine)
    if not cache.available:
        logger.warning("Tesseract not found; pictures will not be OCR'd")

    prs = _open(path)
    chunks = []
    for index, slide in enumerate(prs.slides):
        builder = PageTextBuilder(index + 1, ocr_engine.low_conf_threshold)
        if index == 0:
            builder.hidden.extend(ooxml_hidden_content(path))
            for name in _comment_authors(prs):
                builder.add_hidden("metadata", name, "comment_author")
        if slide._element.get("show") == "0":
            builder.add_hidden("hidden_slide", "Slide is hidden during the slideshow", "slide")

        for shape in _reading_order(slide.shapes):
            try:
                _add_shape(shape, builder, cache)
            except SHAPE_ERRORS as exc:
                builder.errors.append(f"Shape {shape.name!r} could not be read: {exc}")

        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame
            if notes is not None and notes.text.strip():
                for paragraph in notes.paragraphs:
                    builder.add_text(paragraph.text)
                    builder.end_line()
                builder.end_block()
        _add_slide_comments(slide, builder)

        chunks.append(
            ExtractionInput(
                file_id=file_id,
                file_name=path.name,
                source_url=source_url,
                user_id=user_id,
                page_number=index + 1,
                **builder.page_data(),
            )
        )
    return chunks
