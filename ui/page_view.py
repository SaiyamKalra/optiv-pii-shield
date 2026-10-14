import pymupdf
from PIL import Image, ImageDraw

from ocr.ocr_engine import LOW_CONF_THRESHOLD
from redaction.pdf_redactor import PDFRedactor

ENTITY_COLORS = {
    "PERSON": "#8c2f39",
    "EMAIL_ADDRESS": "#1f5f8b",
    "PHONE_NUMBER": "#2e6b3a",
    "STREET_ADDRESS": "#b7791f",
    "DATE_OF_BIRTH": "#a3501f",
    "US_SSN": "#6b4c9a",
    "CREDIT_CARD": "#6b4c9a",
    "PASSPORT": "#6b4c9a",
    "PAN": "#6b4c9a",
    "NATIONAL_ID": "#6b4c9a",
    "EMPLOYEE_ID": "#4f5d75",
    "DIRECTOR_ID": "#4f5d75",
    "VENDOR_ID": "#4f5d75",
    "IP_ADDRESS": "#5b6b73",
}
DEFAULT_COLOR = "#4a4a4a"
UNCERTAIN_RGB = (183, 121, 31)
UNREAD_RGB = (201, 88, 12)


def entity_color(entity_type: str | None) -> str:
    return ENTITY_COLORS.get(entity_type or "", DEFAULT_COLOR)


def _rgb(hex_color: str) -> tuple[int, int, int]:
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rasterize(pdf_bytes: bytes, page_number: int, dpi: int = 110):
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        m = page.rotation_matrix * pymupdf.Matrix(dpi / 72, dpi / 72)
    return img, (m.a, m.b, m.c, m.d, m.e, m.f)


def _rect(box, matrix) -> tuple[float, float, float, float]:
    r = (pymupdf.Rect(box.x, box.y, box.x + box.w, box.y + box.h) * pymupdf.Matrix(*matrix)).normalize()
    return r.x0, r.y0, r.x1, r.y1


def is_uncertain_word(box, threshold: float = LOW_CONF_THRESHOLD) -> bool:
    return (
        box.source == "ocr"
        and box.confidence is not None
        and box.confidence < threshold
        and sum(ch.isalnum() for ch in box.text) >= 2
    )


def page_uncertainty(chunk) -> dict:
    return {
        "low_confidence_words": chunk.low_confidence_word_count,
        "unread_regions": len(chunk.unread_regions),
        "images_skipped": chunk.images_skipped_count,
        "not_inspected": bool(chunk.extraction_errors),
    }


def _hatch(layer: Image.Image, rect, color, step: int = 9):
    x0, y0 = max(0, int(rect[0])), max(0, int(rect[1]))
    x1, y1 = min(layer.width, int(rect[2])), min(layer.height, int(rect[3]))
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0:
        return
    tile = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(tile)
    for k in range(-h, w, step):
        draw.line([(k, h), (k + h, 0)], fill=(*color, 150), width=2)
    layer.alpha_composite(tile, (x0, y0))


def page_overlay(
    img: Image.Image,
    matrix,
    chunk,
    findings=(),
    *,
    findings_layer: bool = True,
    uncertainty_layer: bool = True,
) -> Image.Image:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    if uncertainty_layer:
        for box in chunk.bounding_boxes or []:
            if box.coord_space == "page_pt" and is_uncertain_word(box):
                draw.rectangle(_rect(box, matrix), fill=(*UNCERTAIN_RGB, 70),
                               outline=(*UNCERTAIN_RGB, 230), width=1)
        for region in chunk.unread_regions:
            if region.coord_space != "page_pt":
                continue
            rect = _rect(region, matrix)
            _hatch(layer, rect, UNREAD_RGB)
            draw.rectangle(rect, outline=(*UNREAD_RGB, 255), width=3)

    if findings_layer:
        for finding in findings:
            color = _rgb(entity_color(finding.entity_type))
            for box in PDFRedactor._boxes_for_finding(chunk, finding):
                if box.coord_space != "page_pt":
                    continue
                draw.rectangle(_rect(box, matrix), fill=(*color, 55), outline=(*color, 255), width=2)

    return Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")
