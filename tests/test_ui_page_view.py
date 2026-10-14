import pymupdf
import pytest

from models.schemas import (
    BoundingBox,
    DetectedEntityDetail,
    ExtractionInput,
    UnreadRegion,
)
from ui.page_view import is_uncertain_word, page_overlay, rasterize


def _pdf(rotation=0):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Contact Helena Vorstermann today", fontsize=14)
    words = page.get_text("words")
    if rotation:
        page.set_rotation(rotation)
    data = doc.tobytes()
    doc.close()

    text, boxes = "", []
    for x0, y0, x1, y1, word, *_ in words:
        start = len(text) + (1 if text else 0)
        text = f"{text} {word}" if text else word
        boxes.append(BoundingBox(page=1, text=word, x=x0, y=y0, w=x1 - x0, h=y1 - y0,
                                 char_start=start, char_end=start + len(word)))
    chunk = ExtractionInput(file_id="f", file_name="f.pdf", raw_text=text, bounding_boxes=boxes)
    return data, chunk


def _finding(chunk, value="Helena Vorstermann"):
    start = chunk.raw_text.index(value)
    return DetectedEntityDetail(token="[PERSON_001]", entity_type="PERSON", original_value=value,
                                start=start, end=start + len(value), score=0.9)


def _center(box, matrix):
    point = pymupdf.Point(box.x + box.w / 2, box.y + box.h / 2) * pymupdf.Matrix(*matrix)
    return int(point.x), int(point.y)


@pytest.mark.parametrize("rotation", [0, 90])
def test_findings_layer_marks_the_words_of_the_finding(rotation):
    data, chunk = _pdf(rotation)
    img, matrix = rasterize(data, 1)
    finding = _finding(chunk)
    drawn = page_overlay(img, matrix, chunk, [finding], uncertainty_layer=False)

    helena = next(b for b in chunk.bounding_boxes if b.text == "Helena")
    contact = next(b for b in chunk.bounding_boxes if b.text == "Contact")
    x, y = _center(helena, matrix)
    assert drawn.size == img.size
    changed = sum(
        drawn.getpixel((x + dx, y + dy)) != img.getpixel((x + dx, y + dy))
        for dx in range(-3, 4) for dy in range(-3, 4)
    )
    assert changed > 0
    cx, cy = _center(contact, matrix)
    assert drawn.getpixel((cx, cy)) == img.getpixel((cx, cy))


def test_layers_off_leave_the_page_unchanged():
    data, chunk = _pdf()
    img, matrix = rasterize(data, 1)
    plain = page_overlay(img, matrix, chunk, [_finding(chunk)],
                         findings_layer=False, uncertainty_layer=False)
    assert plain.tobytes() == img.tobytes()


def test_uncertainty_layer_marks_low_confidence_words_and_unread_regions():
    data, chunk = _pdf()
    word = chunk.bounding_boxes[0].model_copy(update={"source": "ocr", "confidence": 30.0})
    region = UnreadRegion(page=1, x=300, y=300, w=60, h=40)
    chunk = chunk.model_copy(update={"bounding_boxes": [word], "unread_regions": [region]})
    img, matrix = rasterize(data, 1)
    drawn = page_overlay(img, matrix, chunk, findings_layer=False)

    x, y = _center(word, matrix)
    assert drawn.getpixel((x, y)) != img.getpixel((x, y))
    rx, ry = _center(region, matrix)
    window = [(rx + d, ry + d) for d in range(-6, 7)]
    assert any(drawn.getpixel(p) != img.getpixel(p) for p in window)


def test_uncertain_word_rule_matches_the_extractor():
    _, chunk = _pdf()
    box = chunk.bounding_boxes[0]
    assert not is_uncertain_word(box)
    assert is_uncertain_word(box.model_copy(update={"source": "ocr", "confidence": 40.0}))
    assert not is_uncertain_word(box.model_copy(update={"source": "ocr", "confidence": 90.0}))
    assert not is_uncertain_word(
        box.model_copy(update={"source": "ocr", "confidence": 10.0, "text": "|"})
    )
