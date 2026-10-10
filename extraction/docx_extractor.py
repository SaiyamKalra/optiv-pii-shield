# extraction/docx_extractor.py

import logging
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import docx
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.ns import qn
from lxml import etree

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

logger = logging.getLogger("DocxExtractor")

MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"
VML_IMAGEDATA = "{urn:schemas-microsoft-com:vml}imagedata"
R_ID = qn("r:id")
R_EMBED = qn("r:embed")

W_P, W_TBL, W_SDT, W_SDT_CONTENT = qn("w:p"), qn("w:tbl"), qn("w:sdt"), qn("w:sdtContent")
W_TR, W_TC = qn("w:tr"), qn("w:tc")
W_T, W_TAB, W_BR, W_CR = qn("w:t"), qn("w:tab"), qn("w:br"), qn("w:cr")
W_LAST_PAGE_BREAK = qn("w:lastRenderedPageBreak")
W_DRAWING, W_PICT, W_TYPE = qn("w:drawing"), qn("w:pict"), qn("w:type")
WP_EXTENT, A_BLIP = qn("wp:extent"), qn("a:blip")
W_PPR, W_PAGE_BREAK_BEFORE, W_VAL = qn("w:pPr"), qn("w:pageBreakBefore"), qn("w:val")
W_DEL_TEXT, W_ID, W_AUTHOR = qn("w:delText"), qn("w:id"), qn("w:author")
W_COMMENT_REF, W_FOOTNOTE_REF, W_ENDNOTE_REF = (
    qn("w:commentReference"), qn("w:footnoteReference"), qn("w:endnoteReference"),
)
W_OBJECT, WP_DOC_PR = qn("w:object"), qn("wp:docPr")
C_CHART = "{http://schemas.openxmlformats.org/drawingml/2006/chart}chart"
DGM_REL_IDS = "{http://schemas.openxmlformats.org/drawingml/2006/diagram}relIds"
R_DM = qn("r:dm")
O_OLE_OBJECT = "{urn:schemas-microsoft-com:office:office}OLEObject"
NOTE_KINDS = {"comments": "comment", "footnotes": "footnote", "endnotes": "endnote"}
# Errors python-docx/lxml raise on malformed content; reported per block, not fatal.
BLOCK_ERRORS = (AttributeError, KeyError, ValueError, TypeError, etree.XMLSyntaxError)


def _load_notes(document) -> dict[str, dict[str, str]]:
    """Comment, footnote and endnote text keyed by kind then id."""
    notes: dict[str, dict[str, str]] = {kind: {} for kind in NOTE_KINDS.values()}
    for rel in document.part.rels.values():
        if rel.is_external:
            continue
        suffix = rel.reltype.rsplit("/", 1)[-1]
        kind = NOTE_KINDS.get(suffix)
        if kind is None:
            continue
        root = etree.fromstring(rel.target_part.blob)
        for note in root:
            text = " ".join(t.text for t in note.iter(W_T) if t.text).strip()
            if not text:
                continue  # Word's separator footnotes have no text
            author = note.get(W_AUTHOR)
            notes[kind][note.get(W_ID)] = f"{author}: {text}" if author else text
    return notes


def _has_page_break(element) -> bool:
    if next(element.iter(W_LAST_PAGE_BREAK), None) is not None:
        return True
    return any(br.get(W_TYPE) == "page" for br in element.iter(W_BR))


def _breaks_before(paragraph) -> bool:
    """Paragraph formatted with 'page break before' (w:pPr/w:pageBreakBefore)."""
    ppr = paragraph.find(W_PPR)
    flag = ppr.find(W_PAGE_BREAK_BEFORE) if ppr is not None else None
    return flag is not None and flag.get(W_VAL) not in ("0", "false", "off")


def _body_images(document) -> list[tuple[str, bytes, float | None]]:
    """(ref, blob, display width) for every large-enough image in the body, for prefetch."""
    part = document.part
    items = {}
    for drawing in document.element.body.iter(W_DRAWING, W_PICT):
        extent = next(drawing.iter(WP_EXTENT), None)
        width_pt = int(extent.get("cx", 0)) / EMU_PER_POINT if extent is not None else None
        height_pt = int(extent.get("cy", 0)) / EMU_PER_POINT if extent is not None else None
        if width_pt is not None and (width_pt < MIN_IMAGE_SIDE_PT or height_pt < MIN_IMAGE_SIDE_PT):
            continue
        rids = [b.get(R_EMBED) for b in drawing.iter(A_BLIP)]
        rids += [v.get(R_ID) for v in drawing.iter(VML_IMAGEDATA)]
        for rid in filter(None, rids):
            image_part = part.related_parts.get(rid)
            if image_part is not None:
                items.setdefault(str(image_part.partname), (image_part.blob, width_pt))
    return [(ref, blob, width) for ref, (blob, width) in items.items()]


def _place_in_image(word: OcrWord):
    return word.left, word.top, word.width, word.height


def _box_in_image(box):
    return box[0], box[1], box[2] - box[0], box[3] - box[1]


class _DocxWalker:
    """Walks the document XML in reading order, splitting pages on Word's saved page breaks.

    DOCX has no fixed pages: Word records where it last broke pages
    (w:lastRenderedPageBreak) and explicit breaks (w:br type="page"). Page numbers
    therefore approximate what Word showed when the file was last saved.
    """

    def __init__(self, document, ocr_engine: OcrEngine, cache: ImageOcrCache):
        self.document = document
        self.threshold = ocr_engine.low_conf_threshold
        self.cache = cache
        self.pages: list[PageTextBuilder] = []
        self.current = PageTextBuilder(1, self.threshold)
        self.buffer: list[str] = []
        self.deleted: list[str] = []
        self.table_depth = 0
        self.notes = _load_notes(document)
        self.notes_used: set[tuple[str, str]] = set()

    # --- page and text buffering -------------------------------------------------
    def flush(self):
        if self.buffer:
            self.current.add_text("".join(self.buffer))
            self.buffer = []
        if self.deleted:
            self.current.add_hidden("deleted_text", "".join(self.deleted), "tracked deletion")
            self.deleted = []

    def add_note(self, kind: str, note_id: str | None):
        text = self.notes[kind].get(note_id)
        if text and (kind, note_id) not in self.notes_used:
            self.notes_used.add((kind, note_id))
            self.current.add_hidden(kind, text, f"{kind} {note_id}")

    def _has_content(self) -> bool:
        page = self.current
        return not page.is_empty or page.images_ocr > 0 or page.images_skipped > 0

    def page_break(self, from_table_row=False):
        if self.table_depth and not from_table_row:
            return  # inside tables, breaks are applied once per row (see walk_table)
        self.flush()
        if not self._has_content():
            return  # consecutive break markers: don't create blank pages
        self.current.end_block()
        self.pages.append(self.current)
        self.current = PageTextBuilder(len(self.pages) + 1, self.threshold)

    def finish(self) -> list[PageTextBuilder]:
        self.flush()
        if self._has_content() or not self.pages:
            self.pages.append(self.current)
        # Notes whose reference wasn't found in the body (e.g. in a header) go to page 1.
        for kind, by_id in self.notes.items():
            for note_id, text in by_id.items():
                if (kind, note_id) not in self.notes_used:
                    self.pages[0].add_hidden(kind, text, f"{kind} {note_id}")
        return self.pages

    # --- block level -------------------------------------------------------------
    def walk_blocks(self, parent, inline=False):
        for child in parent.iterchildren():
            try:
                self.walk_block(child, inline)
            except BLOCK_ERRORS as exc:
                self.buffer = []
                self.current.errors.append(f"Content could not be read: {exc}")

    def walk_block(self, child, inline):
        if child.tag == W_P:
            if _breaks_before(child):
                self.page_break()
            self.walk_inline(child, part=None)
            self.flush()
            if not inline:
                self.current.end_line()
        elif child.tag == W_TBL:
            self.walk_table(child)
        elif child.tag == W_SDT:
            content = child.find(W_SDT_CONTENT)
            if content is not None:
                self.walk_blocks(content, inline)

    def walk_table(self, tbl):
        self.flush()
        self.current.end_block()
        self.table_depth += 1
        previous_row_broke = False
        for row in tbl.iter(W_TR):
            if row.getparent() is not tbl:
                continue  # nested tables are walked through their cell
            # A row split across pages carries break markers in its continuing cells, and
            # Word marks the first cell of the next row too: treat that as one break.
            row_breaks = self.table_depth == 1 and _has_page_break(row)
            if row_breaks and not previous_row_broke:
                self.page_break(from_table_row=True)
            previous_row_broke = row_breaks
            for i, cell in enumerate(c for c in row.iterchildren() if c.tag == W_TC):
                if i:
                    self.current.add_text("|")
                self.walk_blocks(cell, inline=True)
            self.flush()
            self.current.end_line()
        self.table_depth -= 1
        self.current.end_block()

    # --- inside a paragraph ------------------------------------------------------
    def walk_inline(self, element, part):
        for child in element.iterchildren():
            tag = child.tag
            if tag == MC_FALLBACK:
                continue  # legacy duplicate of the mc:Choice content
            if tag == W_T:
                self.buffer.append(child.text or "")
            elif tag == W_DEL_TEXT:
                self.deleted.append(child.text or "")
            elif tag == W_COMMENT_REF:
                self.add_note("comment", child.get(W_ID))
            elif tag == W_FOOTNOTE_REF:
                self.add_note("footnote", child.get(W_ID))
            elif tag == W_ENDNOTE_REF:
                self.add_note("endnote", child.get(W_ID))
            elif tag == W_OBJECT:
                ole = next(child.iter(O_OLE_OBJECT), None)
                name = ole.get("ProgID", "") if ole is not None else ""
                self.current.add_hidden("embedded_object", name or "embedded object", "w:object")
                self.walk_inline(child, part)
            elif tag in (W_TAB, W_CR):
                self.buffer.append(" ")
            elif tag == W_BR:
                if child.get(W_TYPE) == "page":
                    self.page_break()
                else:
                    self.flush()
                    self.current.end_line()
            elif tag == W_LAST_PAGE_BREAK:
                self.page_break()
            elif tag in (W_DRAWING, W_PICT):
                self.flush()
                self.add_drawing_text(child, part)
                self.add_images(child, part)
                self.walk_inline(child, part)  # text boxes inside drawings
            elif tag == W_P:
                self.walk_inline(child, part)  # paragraphs inside text boxes
                self.flush()
                self.current.end_line()
            else:
                self.walk_inline(child, part)

    def add_drawing_text(self, drawing, part):
        """Alt text (hidden) plus chart and SmartArt labels (visible, stored in other parts)."""
        part = part or self.document.part
        doc_pr = next(drawing.iter(WP_DOC_PR), None)
        if doc_pr is not None:
            alt = " ".join(v for v in (doc_pr.get("title"), doc_pr.get("descr")) if v)
            self.current.add_hidden("alt_text", alt, doc_pr.get("name"))
        rids = [c.get(R_ID) for c in drawing.iter(C_CHART)]
        rids += [d.get(R_DM) for d in drawing.iter(DGM_REL_IDS)]
        for rid in filter(None, rids):
            related = part.related_parts.get(rid)
            if related is not None:
                self.current.add_text(drawing_part_text(related.blob))
                self.current.end_block()

    def add_images(self, drawing, part):
        part = part or self.document.part
        extent = next(drawing.iter(WP_EXTENT), None)
        width_pt = None
        if extent is not None:
            width_pt = int(extent.get("cx", 0)) / EMU_PER_POINT
            height_pt = int(extent.get("cy", 0)) / EMU_PER_POINT
            if width_pt < MIN_IMAGE_SIDE_PT or height_pt < MIN_IMAGE_SIDE_PT:
                self.current.images_skipped += 1
                return

        rids = [b.get(R_EMBED) for b in drawing.iter(A_BLIP)]
        rids += [v.get(R_ID) for v in drawing.iter(VML_IMAGEDATA)]
        for rid in filter(None, rids):
            image_part = part.related_parts.get(rid)
            if image_part is None:
                continue
            ref = str(image_part.partname)
            outcome = self.cache.ocr(ref, image_part.blob, display_width_pt=width_pt)
            if outcome.status not in ("ok", "decorative"):
                logger.info("Page %d: image %s not OCR'd (%s)",
                            self.current.page_number, ref, outcome.status)
            # DOCX images have no fixed page position: boxes are image pixels, and an
            # unreadable image is reported as a whole (w = h = 0 means "entire image").
            record_image_outcome(
                self.current, outcome, ref, _place_in_image, _box_in_image,
                coord_space="image_px", whole_image=(0.0, 0.0, 0.0, 0.0),
            )

    # --- headers and footers -----------------------------------------------------
    def walk_headers_footers(self):
        """Add header/footer text once, at the start of page 1 (it repeats on every page)."""
        seen = set()
        for section in self.document.sections:
            for hf in (
                section.header, section.first_page_header, section.even_page_header,
                section.footer, section.first_page_footer, section.even_page_footer,
            ):
                if hf.is_linked_to_previous or id(hf.part) in seen:
                    continue
                seen.add(id(hf.part))
                for p in hf._element.iterchildren(W_P):
                    self.walk_inline(p, part=hf.part)
                    self.flush()
                    self.current.end_line()
        self.current.end_block()


def extract_docx(
    path: str | Path,
    *,
    file_id: str | None = None,
    source_url: str | None = None,
    user_id: str | None = None,
    ocr_engine: OcrEngine | None = None,
) -> list[ExtractionInput]:
    """Extract a DOCX into approximate pages, OCR'ing every embedded image.

    Text, tables, text boxes, headers/footers, charts and SmartArt are read natively
    (no word boxes). OCR words get boxes in image pixels (coord_space="image_px") with
    image_ref naming the image part. Comments, footnotes, endnotes, tracked deletions,
    alt text, document properties and embedded objects go to hidden_content.
    Raises DocumentValidationError if the file isn't a readable DOCX.
    """
    path = Path(path)
    if validate_document(path) != "docx":
        raise DocumentValidationError("unsupported_type", f"{path.name} is not a DOCX")
    file_id = file_id or file_id_for(path)
    source_url = source_url or str(path)
    ocr_engine = ocr_engine or OcrEngine()
    cache = ImageOcrCache(ocr_engine)
    if not cache.available:
        logger.warning("Tesseract not found; images will not be OCR'd")

    try:
        document = docx.Document(str(path))
        walker = _DocxWalker(document, ocr_engine, cache)
    except (PackageNotFoundError, KeyError, ValueError, zipfile.BadZipFile,
            ElementTree.ParseError, etree.XMLSyntaxError) as exc:
        raise DocumentValidationError("corrupt", f"{path.name} can't be read as DOCX: {exc}") from exc
    walker.current.hidden.extend(ooxml_hidden_content(path))
    cache.prefetch(_body_images(document))
    walker.walk_headers_footers()
    walker.walk_blocks(walker.document.element.body)

    return [
        ExtractionInput(
            file_id=file_id,
            file_name=path.name,
            source_url=source_url,
            user_id=user_id,
            page_number=builder.page_number,
            **builder.page_data(),
        )
        for builder in walker.finish()
    ]
