import shutil
import tempfile
from pathlib import Path

from extraction.normalizer import extract_document
from extraction.validation import validate_document
from models.schemas import ExtractionInput, PIIOutput
from pii.pipeline import PIIPipeline
from redaction.docx_redactor import DOCXRedactor
from redaction.office_redactor import image_boxes_to_redact
from redaction.office_xml_utils import (
    MASK_CHAR,
    NER_ENTITY_TYPES,
    pii_values,
    value_pieces,
)
from redaction.pptx_redactor import PPTXRedactor

MAX_PASSES = 2


def scan_chunks(chunks: list[ExtractionInput]) -> list[ExtractionInput]:
    scanned = []
    for chunk in chunks:
        hidden = "\n".join(item.text for item in chunk.hidden_content if item.text)
        text = f"{chunk.raw_text}\n\n{hidden}" if hidden else chunk.raw_text
        scanned.append(chunk.model_copy(update={"raw_text": text}))
    return scanned


def _finding_image_refs(chunks, pii_results) -> set[str]:
    no_unread = [c.model_copy(update={"unread_regions": []}) for c in chunks]
    return set(image_boxes_to_redact(no_unread, pii_results))


def _is_mask_guess(finding) -> bool:
    return finding.entity_type in NER_ENTITY_TYPES and MASK_CHAR in finding.original_value


def _without_mask_guesses(results: list[PIIOutput]) -> list[PIIOutput]:
    return [
        r.model_copy(update={
            "detected_entities": [e for e in r.detected_entities if not _is_mask_guess(e)]
        })
        for r in results
    ]


def _only_ocr(chunks, page_number: int, finding) -> bool:
    chunk = next((c for c in chunks if c.page_number == page_number), None)
    if chunk is None or finding.end > len(chunk.raw_text):
        return False
    covered = set()
    for box in chunk.bounding_boxes or []:
        if box.source == "ocr" and box.char_start is not None and box.char_end is not None:
            covered.update(range(max(box.char_start, finding.start), min(box.char_end, finding.end)))
    return all(
        i in covered or chunk.raw_text[i].isspace() for i in range(finding.start, finding.end)
    )


def _inspect(chunks, label: str) -> tuple[list[dict], list[dict]]:
    issues, warnings = [], []
    for chunk in chunks:
        page = chunk.page_number
        if chunk.extraction_errors:
            issues.append({"page_number": page, "reason": f"{label}: page not fully inspected",
                           "errors": chunk.extraction_errors})
        if chunk.images_skipped_count:
            warnings.append({"page_number": page, "reason": f"{label}: images were skipped",
                             "count": chunk.images_skipped_count})
        if chunk.low_confidence_word_count:
            warnings.append({"page_number": page, "reason": f"{label}: low-confidence OCR words",
                             "count": chunk.low_confidence_word_count})
    return issues, warnings


class OfficeRedactionPipeline:

    def __init__(self, pii_pipeline=None, extractor=extract_document):
        self.pii_pipeline = pii_pipeline
        self.extractor = extractor

    def _detect(self, chunks):
        scanned = scan_chunks(chunks)
        todo = [c for c in scanned if c.raw_text not in self._seen]
        if todo:
            if self.pii_pipeline is None:
                self.pii_pipeline = PIIPipeline()
            for chunk, output in zip(todo, self.pii_pipeline.process_document_chunks(todo)):
                self._seen[chunk.raw_text] = output
        return [
            self._seen[c.raw_text].model_copy(update={"page_number": c.page_number})
            for c in scanned
        ]

    def process(
        self,
        input_path: str,
        output_path: str,
        chunks: list[ExtractionInput] | None = None,
        pii_results: list[PIIOutput] | None = None,
    ) -> dict:
        source, destination = Path(input_path), Path(output_path)
        if source.resolve() == destination.resolve():
            raise ValueError("Output must differ from the original file.")
        if not source.is_file():
            raise FileNotFoundError(source)

        kind = validate_document(source)
        if kind == "docx":
            redactor = DOCXRedactor()
        elif kind == "pptx":
            redactor = PPTXRedactor()
        else:
            raise ValueError("OfficeRedactionPipeline handles DOCX and PPTX only.")

        self._seen: dict[str, PIIOutput] = {}
        if chunks is None:
            chunks = self.extractor(source)
        if pii_results is None:
            pii_results = self._detect(chunks)
        else:
            for chunk, output in zip(scan_chunks(chunks), pii_results):
                self._seen.setdefault(chunk.raw_text, output)

        result = {
            "status": "BLOCKED",
            "output_path": None,
            "redaction_count": 0,
            "unmapped_findings": [],
            "verification_passed": False,
            "inspection_issues": [],
            "inspection_warnings": [],
            "passes": 0,
            "total_pages": len(chunks),
        }

        page_numbers = [chunk.page_number for chunk in chunks]
        if not chunks or page_numbers != list(range(1, len(chunks) + 1)):
            result["inspection_issues"].append(
                {"reason": "Extraction does not cover every page/slide exactly once",
                 "received_pages": page_numbers}
            )
            return result

        issues, warnings = _inspect(chunks, "original")
        destination.unlink(missing_ok=True)

        current_src, current_chunks, current_results = source, chunks, pii_results
        masked: set[str] = set()
        rescanned_results = None
        with tempfile.TemporaryDirectory() as work:
            try:
                for attempt in range(1, MAX_PASSES + 1):
                    out = Path(work) / f"pass{attempt}{source.suffix}"
                    stats = redactor.redact(str(current_src), str(out), current_chunks, current_results)
                    masked |= {v.lower() for v in pii_values(current_results)}
                    result["passes"] = attempt
                    result["redaction_count"] += stats["text_replacements"] + stats["images_redacted"]

                    finding_refs = _finding_image_refs(current_chunks, current_results)
                    for ref, error in stats["images_failed"].items():
                        entry = {"reason": f"Could not redact image {ref}", "error": error}
                        (issues if ref in finding_refs else warnings).append(entry)

                    rescanned = self.extractor(out)
                    if [c.page_number for c in rescanned] != page_numbers:
                        raise ValueError("Rescan did not cover the same pages/slides")
                    rescanned_results = self._detect(rescanned)
                    current_src, current_chunks = out, rescanned
                    current_results = _without_mask_guesses(rescanned_results)
                    if not any(r.detected_entities for r in current_results):
                        break

                shutil.copyfile(current_src, destination)
            except (OSError, ValueError, RuntimeError) as exc:
                issues.append({"reason": "Office redaction or rescan failed", "error": str(exc)})
                destination.unlink(missing_ok=True)

        if rescanned_results is not None and destination.is_file():
            rescan_issues, rescan_warnings = _inspect(current_chunks, "rescan")
            issues += rescan_issues
            warnings += rescan_warnings
            result["verification_passed"] = len(rescanned_results) == len(chunks) and all(
                r.verification_passed and not r.detected_entities for r in rescanned_results
            )
            leaks, guesses = [], []
            for r in rescanned_results:
                for e in r.detected_entities:
                    entry = {"page_number": r.page_number, "entity_type": e.entity_type}
                    already_masked = any(p.lower() in masked for p in value_pieces(e.original_value))
                    if e.entity_type not in NER_ENTITY_TYPES or (
                        already_masked
                        and not _is_mask_guess(e)
                        and not _only_ocr(current_chunks, r.page_number, e)
                    ):
                        leaks.append(entry)
                    else:
                        guesses.append(entry)
            if leaks:
                issues.append({"reason": f"Rescan still finds PII after {result['passes']} passes",
                               "findings": leaks})
            if guesses:
                warnings.append({"reason": "Rescan: name/place/organisation guesses on masked "
                                           "text; review", "findings": guesses})
            result["verification_passed"] = result["verification_passed"] or (
                not leaks and len(rescanned_results) == len(chunks)
                and all(r.verification_passed for r in rescanned_results)
            )

        if issues or not result["verification_passed"]:
            status = "BLOCKED"
        elif warnings:
            status = "REVIEW_REQUIRED"
        else:
            status = "VERIFIED_NO_PII_FOUND"

        result.update(
            status=status,
            output_path=str(destination) if destination.is_file() else None,
            inspection_issues=issues,
            inspection_warnings=warnings,
        )
        return result
