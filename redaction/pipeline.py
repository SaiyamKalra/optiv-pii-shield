from pathlib import Path
import pymupdf
from models.schemas import ExtractionInput, PIIOutput
from redaction.pdf_redactor import PDFRedactor
from redaction.verifier import RedactionVerifier
from extraction.pdf_extractor import extract_pdf
from pii.pipeline import PIIPipeline


class RedactionPipeline:

    @staticmethod
    def _remove_output(output_path):
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass

    def __init__(self, extractor=extract_pdf, pii_pipeline=None):
        self.redactor = PDFRedactor()
        self.verifier = RedactionVerifier()
        self.extractor = extractor
        self.pii_pipeline = pii_pipeline

    def process(
        self,
        input_pdf: str,
        output_pdf: str,
        chunks: list[ExtractionInput],
        pii_results: list[PIIOutput],
    ) -> dict:

        input_path = Path(input_pdf)
        output_path = Path(output_pdf)

        if input_path.resolve() == output_path.resolve():
            raise ValueError("Output PDF must be different from the original PDF.")

        if not input_path.is_file():
            raise FileNotFoundError(f"Input PDF not found: {input_pdf}")

        with pymupdf.open(input_path) as document:
            actual_page_count = len(document)

        chunk_pages = [chunk.page_number for chunk in chunks]
        expected_pages = set(range(1, actual_page_count + 1))

        if (
            len(chunk_pages) != actual_page_count
            or set(chunk_pages) != expected_pages
            or len(set(chunk_pages)) != len(chunk_pages)
        ):
            return {
                "status": "BLOCKED",
                "output_pdf": None,
                "redaction_count": 0,
                "unmapped_findings": [],
                "verification_passed": False,
                "inspection_issues": [
                    {
                        "reason": "Extraction does not cover every PDF page",
                        "expected_pages": sorted(expected_pages),
                        "received_pages": chunk_pages,
                    }
                ],
                "inspection_warnings": [],
            }

        inspection_issues = []
        inspection_warnings = []

        for chunk in chunks:
            page = chunk.page_number

            used_ocr = chunk.ocr_used or chunk.extraction_method in ("ocr", "hybrid")

            if used_ocr and chunk.ocr_mean_confidence is None:
                inspection_issues.append(
                    {
                        "page_number": page,
                        "reason": "OCR confidence is unavailable",
                    }
                )

            if chunk.low_confidence_word_count > 0:
                inspection_issues.append(
                    {
                        "page_number": page,
                        "reason": "OCR contains low-confidence words",
                        "count": chunk.low_confidence_word_count,
                    }
                )

            if chunk.images_skipped_count > 0:
                inspection_warnings.append(
                    {
                        "page_number": page,
                        "reason": "Some images were skipped during extraction",
                        "count": chunk.images_skipped_count,
                    }
                )

        if inspection_issues:
            return {
                "status": "BLOCKED",
                "output_pdf": None,
                "redaction_count": 0,
                "unmapped_findings": [],
                "verification_passed": False,
                "inspection_issues": inspection_issues,
                "inspection_warnings": inspection_warnings,
            }

        unmapped_findings = []

        for result in pii_results:
            chunk = next(
                (item for item in chunks if item.page_number == result.page_number),
                None,
            )

            for finding in result.detected_entities:
                boxes = (
                    self.redactor._boxes_for_finding(chunk, finding)
                    if chunk is not None
                    else []
                )

                if not boxes:
                    unmapped_findings.append(
                        {
                            "page_number": result.page_number,
                            "entity_type": finding.entity_type,
                            "start": finding.start,
                            "end": finding.end,
                        }
                    )
        if unmapped_findings:
            return {
                "status": "BLOCKED",
                "output_pdf": None,
                "redaction_count": 0,
                "unmapped_findings": unmapped_findings,
                "verification_passed": False,
                "inspection_issues": [
                    {
                        "reason": (
                            "One or more PII findings cannot be mapped "
                            "to PDF locations"
                        )
                    }
                ],
                "inspection_warnings": inspection_warnings,
            }

        try:
            redaction_count = self.redactor.redact(
                input_pdf=input_pdf,
                output_pdf=output_pdf,
                chunks=chunks,
                pii_results=pii_results,
            )

            rescanned_chunks = self.extractor(output_path)

            rescanned_pages = [chunk.page_number for chunk in rescanned_chunks]
            expected_pages = set(range(1, actual_page_count + 1))

            if (
                len(rescanned_pages) != actual_page_count
                or set(rescanned_pages) != expected_pages
                or len(set(rescanned_pages)) != len(rescanned_pages)
            ):
                self._remove_output(output_path)
                return {
                    "status": "BLOCKED",
                    "output_pdf": None,
                    "redaction_count": redaction_count,
                    "unmapped_findings": [],
                    "verification_passed": False,
                    "inspection_issues": [
                        {"reason": "Fresh extraction did not cover every output page"}
                    ],
                    "inspection_warnings": inspection_warnings,
                }

            for chunk in rescanned_chunks:
                used_ocr = chunk.ocr_used or chunk.extraction_method in (
                    "ocr",
                    "hybrid",
                )

                if used_ocr and chunk.ocr_mean_confidence is None:
                    inspection_issues.append(
                        {
                            "page_number": chunk.page_number,
                            "reason": "Rescan OCR confidence is unavailable",
                        }
                    )

                if chunk.low_confidence_word_count > 0:
                    inspection_issues.append(
                        {
                            "page_number": chunk.page_number,
                            "reason": "Rescan contains low-confidence OCR words",
                            "count": chunk.low_confidence_word_count,
                        }
                    )

                if chunk.images_skipped_count > 0:
                    inspection_warnings.append(
                        {
                            "page_number": chunk.page_number,
                            "reason": "Images were skipped during rescan",
                            "count": chunk.images_skipped_count,
                        }
                    )

            if self.pii_pipeline is None:
                self.pii_pipeline = PIIPipeline()

            rescanned_results = self.pii_pipeline.process_document_chunks(
                rescanned_chunks
            )

            verification_passed = self.verifier.verify(
                rescanned_results,
                expected_pages=actual_page_count,
            )

        except Exception as exc:
            self._remove_output(output_path)
            return {
                "status": "BLOCKED",
                "output_pdf": None,
                "redaction_count": redaction_count,
                "unmapped_findings": [],
                "verification_passed": False,
                "inspection_issues": [
                    {
                        "reason": "Fresh extraction or PII rescan failed",
                        "error": str(exc),
                    }
                ],
                "inspection_warnings": inspection_warnings,
            }

        if inspection_issues or not verification_passed:
            status = "BLOCKED"
        elif inspection_warnings:
            status = "REVIEW_REQUIRED"
        else:
            status = "VERIFIED_NO_PII_FOUND"

        if status != "VERIFIED_NO_PII_FOUND":
            self._remove_output(output_path)

        return {
            "status": status,
            "output_pdf": (
                str(output_path) if status == "VERIFIED_NO_PII_FOUND" else None
            ),
            "redaction_count": redaction_count,
            "unmapped_findings": unmapped_findings,
            "verification_passed": verification_passed,
            "inspection_issues": inspection_issues,
            "inspection_warnings": inspection_warnings,
        }
