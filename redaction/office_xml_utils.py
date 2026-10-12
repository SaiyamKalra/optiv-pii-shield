"""Redact a DOCX/PPTX package: mask PII text in its XML parts and black out PII
(and ink OCR could not read) inside its embedded images.

lxml is used instead of ElementTree because ElementTree renames namespace prefixes
and drops unused declarations, which breaks mc:Ignorable and makes Word/PowerPoint
report the file as corrupt.
"""

import io
import posixpath
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from lxml import etree
from PIL import Image, ImageDraw

MASK_CHAR = "█"
MIN_VALUE_LEN = 3  # shorter "values" (e.g. a lone initial) would wreck the document
IMAGE_PAD_PX = 2
# Text nodes that hold visible run text, joined per paragraph so PII split across
# runs (e.g. "John" in bold + " Smith") is still found.
RUN_TEXT_TAGS = {"t", "delText", "instrText"}
BREAK_TAGS = {"br", "cr", "tab"}
TEXT_ATTRIBUTES = {"descr", "title"}  # alt text on pictures and shapes
SAVABLE_IMAGE_FORMATS = {"PNG", "JPEG", "GIF", "BMP", "TIFF", "WEBP"}
CONTENT_TYPES = "[Content_Types].xml"
# Model-based entity types; anything else (regex recognizers) is a structured identifier.
NER_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "NRP", "GPE", "DATE_TIME", "MISC"}

_PARSER = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=True)


def _local(tag) -> str:
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def value_pieces(value: str) -> list[str]:
    """Split a detected value where extraction joined separate XML elements: table
    cells (" | ") and lines/paragraphs. Each piece is matched on its own."""
    pieces = (" ".join(piece.split()) for piece in re.split(r"\s*\|\s*|\n", str(value)))
    return [piece for piece in pieces if len(piece) >= MIN_VALUE_LEN]


def pii_terms(pii_results) -> list[tuple[str, bool]]:
    """(value piece, ignore_case) to mask, longest first. NER values match case-
    sensitively: one bogus "AND" ORGANIZATION must not blank every "and" in the file.
    Structured identifiers (emails, IDs) match in any case."""
    terms: dict[str, bool] = {}
    for result in pii_results:
        for entity in result.detected_entities:
            ignore_case = entity.entity_type not in NER_ENTITY_TYPES
            for piece in value_pieces(entity.original_value or ""):
                terms[piece] = terms.get(piece, False) or ignore_case
    return sorted(terms.items(), key=lambda t: len(t[0]), reverse=True)


def pii_values(pii_results) -> list[str]:
    """Distinct detected values (split into pieces), longest first."""
    values = {
        piece
        for result in pii_results
        for entity in result.detected_entities
        if entity.original_value
        for piece in value_pieces(entity.original_value)
    }
    return sorted(values, key=len, reverse=True)


class ValueMatcher:
    """Finds detected values in text. Case-insensitive; whitespace inside a value
    matches any whitespace (raw_text joins words with single spaces); a value never
    matches inside a longer word. Each value is searched on its own, so overlapping
    values ("Retain the" and "the Vendor") are both found."""

    def __init__(self, values: list[str | tuple[str, bool]]):
        self.patterns = []
        for value in values:
            text, ignore_case = (value, True) if isinstance(value, str) else value
            self.patterns.append(re.compile(
                r"(?<!\w)" + r"\s+".join(map(re.escape, text.split())) + r"(?!\w)",
                re.IGNORECASE if ignore_case else 0,
            ))

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for pattern in self.patterns for m in pattern.finditer(text)]


def value_pattern(values: list[str | tuple[str, bool]]) -> ValueMatcher | None:
    """Plain strings match in any case; (value, ignore_case) pairs as given."""
    return ValueMatcher(values) if values else None


def mask_text(text: str | None, matcher: ValueMatcher | None) -> tuple[str | None, int]:
    """Replace every match with block characters of the same length."""
    if not text or matcher is None:
        return text, 0
    spans = matcher.spans(text)
    if not spans:
        return text, 0
    chars = list(text)
    for start, end in spans:
        chars[start:end] = MASK_CHAR * (end - start)
    return "".join(chars), len(spans)


def _serialize(tree) -> bytes:
    docinfo = tree.docinfo
    return etree.tostring(
        tree,
        xml_declaration=True,
        encoding=docinfo.encoding or "UTF-8",
        standalone=docinfo.standalone,
    )


def redact_xml(data: bytes, pattern: ValueMatcher | None) -> tuple[bytes, int]:
    """Mask PII in one XML part. Returns the original bytes if nothing changed."""
    tree = etree.parse(io.BytesIO(data), _PARSER)
    root = tree.getroot()
    count = 0

    # Run text of the whole part in document order, "\n" between paragraphs. Extraction
    # joins runs, paragraphs and table cells with spaces, so a detected value can span
    # several elements (e.g. "T. Lindqvist" + "CIO" in two paragraphs of one cell);
    # the pattern's \s+ matches across those breaks.
    # Line breaks and tabs (<w:br/>, <a:br/>, <w:tab/>) hold no text but separate words.
    nodes, last_paragraph = [], None
    for node in root.iter():
        name = _local(node.tag)
        if name in BREAK_TAGS:
            if nodes and nodes[-1] is not None:
                nodes.append(None)
            continue
        if name not in RUN_TEXT_TAGS or not node.text:
            continue
        paragraph = next((a for a in node.iterancestors() if _local(a.tag) == "p"), None)
        if nodes and nodes[-1] is not None and paragraph is not last_paragraph:
            nodes.append(None)  # paragraph break
        nodes.append(node)
        last_paragraph = paragraph
    if nodes:
        combined = "".join("\n" if n is None else n.text for n in nodes)
        masked, replaced = mask_text(combined, pattern)
        if replaced:
            count += replaced
            position = 0
            for node in nodes:
                if node is None:
                    position += 1
                    continue
                node.text = masked[position : position + len(node.text)]
                position += len(node.text)

    external_rel = _local(root.tag) == "Relationships"
    for node in root.iter():
        if not isinstance(node.tag, str):  # comments, processing instructions
            continue
        node.text, replaced = mask_text(node.text, pattern)
        count += replaced
        node.tail, replaced = mask_text(node.tail, pattern)
        count += replaced
        for key, value in list(node.attrib.items()):
            name = _local(key)
            if name in TEXT_ATTRIBUTES or (
                external_rel and name == "Target" and node.get("TargetMode") == "External"
            ):
                masked, replaced = mask_text(value, pattern)
                if replaced:
                    node.set(key, masked)
                    count += replaced

    if not count:
        return data, 0
    return _serialize(tree), count


def redact_image(
    data: bytes, boxes: list[tuple[float, float, float, float] | None]
) -> tuple[bytes, str]:
    """Black out pixel boxes (x, y, w, h); None means the whole image.

    Returns (bytes, format). Formats Pillow can read but not write (EMF/WMF) come back
    as PNG, rendered at the same size the OCR saw, so the boxes still line up.
    """
    img = Image.open(io.BytesIO(data))
    img.load()
    fmt = img.format if img.format in SAVABLE_IMAGE_FORMATS else "PNG"

    if fmt == "JPEG" or img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA" if fmt != "JPEG" and "A" in img.getbands() else "RGB")
    fill = 0 if img.mode == "L" else (0, 0, 0, 255) if img.mode == "RGBA" else (0, 0, 0)

    draw = ImageDraw.Draw(img)
    for box in boxes:
        if box is None:
            draw.rectangle((0, 0, img.width, img.height), fill=fill)
            continue
        x, y, w, h = box
        draw.rectangle(
            (
                max(0, x - IMAGE_PAD_PX),
                max(0, y - IMAGE_PAD_PX),
                min(img.width, x + w + IMAGE_PAD_PX),
                min(img.height, y + h + IMAGE_PAD_PX),
            ),
            fill=fill,
        )

    out = io.BytesIO()
    if fmt == "JPEG":
        img.save(out, format=fmt, quality=95)
    elif fmt == "GIF":
        img.convert("RGB").save(out, format=fmt)
    else:
        img.save(out, format=fmt)
    return out.getvalue(), fmt


def _renamed_part(name: str, taken: set[str]) -> str:
    stem = posixpath.splitext(name)[0]
    candidate, n = f"{stem}_redacted.png", 1
    while candidate in taken:
        n += 1
        candidate = f"{stem}_redacted{n}.png"
    return candidate


def _retarget_rels(data: bytes, rels_name: str, renames: dict[str, str]) -> bytes:
    """Point relationships at renamed parts (e.g. image1.emf -> image1_redacted.png)."""
    tree = etree.parse(io.BytesIO(data), _PARSER)
    # word/_rels/document.xml.rels holds targets relative to word/
    base = posixpath.dirname(posixpath.dirname(rels_name))
    changed = False
    for rel in tree.getroot():
        target = rel.get("Target")
        if not target or rel.get("TargetMode") == "External":
            continue
        absolute = target.startswith("/")
        resolved = target.lstrip("/") if absolute else posixpath.normpath(posixpath.join(base, target))
        if resolved in renames:
            new = renames[resolved]
            rel.set("Target", "/" + new if absolute else posixpath.relpath(new, base or "."))
            changed = True
    return _serialize(tree) if changed else data


def _content_types_with_png(data: bytes, renames: dict[str, str]) -> bytes:
    """Declare PNG and drop overrides for parts that were renamed."""
    tree = etree.parse(io.BytesIO(data), _PARSER)
    root = tree.getroot()
    ns = root.nsmap.get(None)
    for child in list(root):
        if _local(child.tag) == "Override" and child.get("PartName", "").lstrip("/") in renames:
            root.remove(child)
    if not any(
        _local(c.tag) == "Default" and (c.get("Extension") or "").lower() == "png" for c in root
    ):
        default = etree.Element(f"{{{ns}}}Default" if ns else "Default")
        default.set("Extension", "png")
        default.set("ContentType", "image/png")
        root.insert(0, default)
    return _serialize(tree)


def _zipinfo(item: ZipInfo, name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=item.date_time)
    info.compress_type = ZIP_DEFLATED
    return info


def redact_office_archive(
    input_path,
    output_path,
    values: list[str | tuple[str, bool]],
    image_boxes: dict[str, list[tuple[float, float, float, float] | None]],
) -> dict:
    """Write a redacted copy of the package.

    image_boxes maps an image part name (e.g. '/word/media/image3.png') to the pixel
    boxes to black out. Returns counts plus the image parts that could not be edited.
    """
    pattern = value_pattern(values)
    wanted_images = {ref.lstrip("/"): boxes for ref, boxes in image_boxes.items() if boxes}
    text_replacements = 0
    images_failed: dict[str, str] = {}

    with ZipFile(input_path, "r") as original:
        bad_part = original.testzip()
        if bad_part:
            raise ValueError(f"Corrupt Office package part: {bad_part}")
        names = original.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Office package contains duplicate part names.")
        parts = [(item, original.read(item.filename)) for item in original.infolist()]

    # Images first: one Pillow can read but not write is saved as PNG under a new name.
    renames: dict[str, str] = {}
    new_images: dict[str, bytes] = {}
    for item, data in parts:
        if item.filename not in wanted_images:
            continue
        try:
            new_images[item.filename], fmt = redact_image(data, wanted_images[item.filename])
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            images_failed["/" + item.filename] = str(exc) or type(exc).__name__
            continue
        if fmt == "PNG" and not item.filename.lower().endswith(".png"):
            renames[item.filename] = _renamed_part(item.filename, set(names) | set(renames.values()))
    for name in set(wanted_images) - set(names):
        images_failed["/" + name] = "image part not found in package"

    with ZipFile(output_path, "w", ZIP_DEFLATED) as cleaned:
        for item, data in parts:
            name = item.filename
            if name in new_images:
                data = new_images[name]
            elif name == CONTENT_TYPES:
                if renames:
                    data = _content_types_with_png(data, renames)
            elif name.endswith((".xml", ".rels")):
                try:
                    data, replaced = redact_xml(data, pattern)
                    if renames and name.endswith(".rels"):
                        data = _retarget_rels(data, name, renames)
                except etree.XMLSyntaxError as exc:
                    raise ValueError(f"Could not inspect Office part: {name}") from exc
                text_replacements += replaced
            cleaned.writestr(_zipinfo(item, renames[name]) if name in renames else item, data)

    with ZipFile(output_path, "r") as check:
        bad_part = check.testzip()
        if bad_part:
            raise ValueError(f"Output package is corrupt: {bad_part}")

    return {
        "text_replacements": text_replacements,
        "images_redacted": len(new_images),
        "images_failed": images_failed,
        "renamed_images": {"/" + old: "/" + new for old, new in renames.items()},
    }
