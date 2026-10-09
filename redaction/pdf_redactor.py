import pymupdf
from models.schemas import ExtractionInput, PIIOutput


class PDFRedactor:

    @staticmethod
    def _boxes_for_finding(chunk, finding):
        if not chunk.bounding_boxes:
            return []

        return [
            box
            for box in chunk.bounding_boxes
            if (
                box.char_start is not None
                and box.char_end is not None
                and box.char_start < finding.end
                and box.char_end > finding.start
            )
        ]

    def redact(
        self,
        input_pdf: str,
        output_pdf: str,
        chunks: list[ExtractionInput],
        pii_results: list[PIIOutput],
    ) -> int:

        document = pymupdf.open(input_pdf)
        redaction_count = 0

        try:
            for result in pii_results:

                chunk = next(
                    (
                        chunk
                        for chunk in chunks
                        if chunk.page_number == result.page_number
                    ),
                    None,
                )

                if chunk is None:
                    raise ValueError(
                        f"Missing extraction chunk for page {result.page_number}"
                    )

                page_index = result.page_number - 1

                if page_index < 0 or page_index >= len(document):
                    raise ValueError(f"Invalid PDF page number: {result.page_number}")

                page = document[page_index]

                for finding in result.detected_entities:

                    boxes = self._boxes_for_finding(
                        chunk,
                        finding,
                    )

                    for box in boxes:

                        rect = pymupdf.Rect(
                            box.x,
                            box.y,
                            box.x + box.w,
                            box.y + box.h,
                        )

                        page.add_redact_annot(
                            rect,
                            fill=(0, 0, 0),
                        )

                        redaction_count += 1

                page.apply_redactions()

            document.save(output_pdf)

        finally:
            document.close()

        return redaction_count
