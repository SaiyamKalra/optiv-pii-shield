from models.schemas import ExtractionInput, PIIOutput
from redaction.office_xml_utils import pii_terms, redact_office_archive, value_pattern


def _box_in_image(box):
    """Pixel box inside the image, or None if the word has no image position."""
    if box.image_box is not None:
        return tuple(box.image_box)
    if box.coord_space == "image_px":
        return (box.x, box.y, box.w, box.h)
    return None


def image_boxes_to_redact(
    chunks: list[ExtractionInput], pii_results: list[PIIOutput]
) -> dict[str, list]:
    """Image part -> pixel boxes to black out. None = the whole image.

    - OCR words inside a finding;
    - OCR words matching any detected value elsewhere (text is masked by value, so
      images are too: the detector may flag only some occurrences);
    - every unread region (ink OCR could not read may hold PII).
    """
    by_page = {chunk.page_number: chunk for chunk in chunks}
    spans: dict[int, list[tuple[int, int]]] = {}
    for result in pii_results:
        spans.setdefault(result.page_number, []).extend(
            (f.start, f.end) for f in result.detected_entities
        )
    pattern = value_pattern(pii_terms(pii_results))
    if pattern is not None:
        for chunk in chunks:
            spans.setdefault(chunk.page_number, []).extend(
                pattern.spans(chunk.raw_text)
            )

    targets: dict[str, list] = {}
    for page, page_spans in spans.items():
        chunk = by_page.get(page)
        if chunk is None:
            continue
        for box in chunk.bounding_boxes or []:
            if not box.image_ref or box.char_start is None or box.char_end is None:
                continue
            if any(box.char_start < end and box.char_end > start for start, end in page_spans):
                pixel_box = _box_in_image(box)
                if pixel_box is not None:
                    targets.setdefault(box.image_ref, []).append(pixel_box)

    for chunk in chunks:
        for region in chunk.unread_regions:
            if not region.image_ref:
                continue
            if region.image_box is not None:
                pixel_box = tuple(region.image_box)
            elif region.coord_space == "image_px" and (region.w or region.h):
                pixel_box = (region.x, region.y, region.w, region.h)
            else:
                pixel_box = None  # unreadable image: black out all of it
            targets.setdefault(region.image_ref, []).append(pixel_box)

    return targets


class OfficeRedactor:
    """Redacts a DOCX/PPTX package. Text is matched by value (Office text has no
    fixed position); image text is blacked out by the OCR word boxes."""

    def redact(
        self,
        input_path: str,
        output_path: str,
        chunks: list[ExtractionInput],
        pii_results: list[PIIOutput],
    ) -> dict:
        return redact_office_archive(
            input_path,
            output_path,
            pii_terms(pii_results),
            image_boxes_to_redact(chunks, pii_results),
        )
