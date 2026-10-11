from pathlib import Path

import pymupdf

from extraction.pdf_extractor import extract_pdf
from models.schemas import ExtractionInput, PIIOutput
from pii.pipeline import PIIPipeline
from redaction.pdf_redactor import PDFRedactor
from redaction.verifier import RedactionVerifier


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

    def _extract_with_retry(self, path):
        """Retry extraction at 400 DPI; support simple test extractors."""
        try:
            return self.extractor(path, dpi=400)
        except TypeError as exc:
            if "dpi" not in str(exc):
                raise
            return self.extractor(path)

    @staticmethod
    def _inspect_chunks(chunks):
        issues = []
        warnings = []

        for chunk in chunks:
            page = chunk.page_number
            used_ocr = chunk.ocr_used or chunk.extraction_method in ("ocr", "hybrid")

            if used_ocr and chunk.ocr_mean_confidence is None:
                issues.append(
                    {
                        "page_number": page,
                        "reason": "OCR confidence is unavailable",
                    }
                )

            if chunk.low_confidence_word_count > 0:
                issues.append(
                    {
                        "page_number": page,
                        "reason": "OCR contains low-confidence words",
                        "count": chunk.low_confidence_word_count,
                    }
                )

            if chunk.images_skipped_count > 0:
                warnings.append(
                    {
                        "page_number": page,
                        "reason": "Some images were skipped during extraction",
                        "count": chunk.images_skipped_count,
                    }
                )

        return issues, warnings

    @staticmethod
    def _coverage(chunks, total_pages, issues, warnings):
        issue_pages = {
            item["page_number"]
            for item in issues
            if item.get("page_number") is not None
        }
        warning_pages = {
            item["page_number"]
            for item in warnings
            if item.get("page_number") is not None
        }

        document_level_issue = any(item.get("page_number") is None for item in issues)
        document_level_warning = any(
            item.get("page_number") is None for item in warnings
        )

        if document_level_issue or document_level_warning:
            reliable_pages = []
        else:
            reliable_pages = sorted(
                {
                    chunk.page_number
                    for chunk in chunks
                    if chunk.page_number not in issue_pages
                    and chunk.page_number not in warning_pages
                }
            )

        coverage = len(reliable_pages) / total_pages * 100 if total_pages else 0.0

        return reliable_pages, coverage

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
            try:
                retry_chunks = self._extract_with_retry(input_path)
                retry_pages = [chunk.page_number for chunk in retry_chunks]

                if (
                    len(retry_pages) != actual_page_count
                    or set(retry_pages) != expected_pages
                    or len(set(retry_pages)) != len(retry_pages)
                ):
                    raise ValueError("Retry extraction still does not cover every page")

                chunks = retry_chunks

                if self.pii_pipeline is None:
                    self.pii_pipeline = PIIPipeline()

                pii_results = self.pii_pipeline.process_document_chunks(chunks)

            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                self._remove_output(output_path)
                return {
                    "status": "BLOCKED",
                    "output_pdf": None,
                    "redaction_count": 0,
                    "unmapped_findings": [],
                    "verification_passed": False,
                    "inspection_issues": [
                        {
                            "reason": "Page coverage failed after retry",
                            "error": str(exc),
                            "expected_pages": sorted(expected_pages),
                            "received_pages": [chunk.page_number for chunk in chunks],
                        }
                    ],
                    "inspection_warnings": [],
                    "total_pages": actual_page_count,
                    "reliable_pages": 0,
                    "reliable_page_numbers": [],
                    "inspection_coverage": 0.0,
                }

        inspection_issues = []
        inspection_warnings = []

        inspection_issues, inspection_warnings = self._inspect_chunks(chunks)

        original_inspection_warnings = list(inspection_warnings)


        if inspection_issues or inspection_warnings:
            try:
                retry_chunks = self._extract_with_retry(input_path)
                retry_pages = [chunk.page_number for chunk in retry_chunks]

                if (
                    len(retry_pages) == actual_page_count
                    and set(retry_pages) == expected_pages
                    and len(set(retry_pages)) == len(retry_pages)
                ):
                    chunks = retry_chunks

                    if self.pii_pipeline is None:
                        self.pii_pipeline = PIIPipeline()

                    pii_results = self.pii_pipeline.process_document_chunks(chunks)

                    inspection_issues, retry_warnings = self._inspect_chunks(chunks)
                    inspection_warnings = (
                        original_inspection_warnings + retry_warnings
                        )
                else:
                    inspection_issues = [
                        {
                            "reason": "Retry extraction did not cover every page",
                            "expected_pages": sorted(expected_pages),
                            "received_pages": retry_pages,
                        }
                    ]

            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                inspection_issues = [
                    {
                        "reason": "Retry extraction failed",
                        "error": str(exc),
                    }
                ]

        if inspection_issues:
            reliable_page_numbers, inspection_coverage = self._coverage(
                chunks,
                actual_page_count,
                inspection_issues,
                inspection_warnings,
            )
            self._remove_output(output_path)
            return {
                "status": "BLOCKED",
                "output_pdf": None,
                "redaction_count": 0,
                "unmapped_findings": [],
                "verification_passed": False,
                "inspection_issues": inspection_issues,
                "inspection_warnings": inspection_warnings,
                "total_pages": actual_page_count,
                "reliable_pages": len(reliable_page_numbers),
                "reliable_page_numbers": reliable_page_numbers,
                "inspection_coverage": inspection_coverage,
            }

        original_inspection_warnings = list(inspection_warnings)

        def find_unmapped(current_chunks, current_results):
            unmapped = []

            for result in current_results:
                chunk = next(
                    (
                        item
                        for item in current_chunks
                        if item.page_number == result.page_number
                    ),
                    None,
                )

                for finding in result.detected_entities:
                    boxes = (
                        self.redactor._boxes_for_finding(chunk, finding)
                        if chunk is not None
                        else []
                    )

                    if not boxes:
                        unmapped.append(
                            {
                                "page_number": result.page_number,
                                "entity_type": finding.entity_type,
                                "start": finding.start,
                                "end": finding.end,
                            }
                        )

            return unmapped

        unmapped_findings = find_unmapped(chunks, pii_results)

        if unmapped_findings:
            try:
                retry_chunks = self._extract_with_retry(input_path)
                retry_pages = [chunk.page_number for chunk in retry_chunks]

                if (
                    len(retry_pages) == actual_page_count
                    and set(retry_pages) == expected_pages
                    and len(set(retry_pages)) == len(retry_pages)
                ):
                    original_unmapped = find_unmapped(retry_chunks, pii_results)

                    if self.pii_pipeline is None:
                        self.pii_pipeline = PIIPipeline()

                    retry_results = self.pii_pipeline.process_document_chunks(
                        retry_chunks
                    )
                    retry_unmapped = find_unmapped(retry_chunks, retry_results)

                    chunks = retry_chunks
                    pii_results = retry_results
                    unmapped_findings = original_unmapped + retry_unmapped

                    inspection_issues, inspection_warnings = self._inspect_chunks(
                        chunks
                    )
                else:
                    unmapped_findings = unmapped_findings + [
                        {"reason": "Retry extraction did not cover every page"}
                    ]

            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                unmapped_findings = unmapped_findings + [
                    {
                        "reason": "Retry extraction for unmapped PII failed",
                        "error": str(exc),
                    }
                ]

        if unmapped_findings:
            reliable_page_numbers, inspection_coverage = self._coverage(
                chunks,
                actual_page_count,
                inspection_issues,
                inspection_warnings,
            )
            self._remove_output(output_path)
            return {
                "status": "BLOCKED",
                "output_pdf": None,
                "redaction_count": 0,
                "unmapped_findings": unmapped_findings,
                "verification_passed": False,
                "inspection_issues": [
                    {"reason": "PII findings remain unmapped after retry"}
                ],
                "inspection_warnings": inspection_warnings,
                "total_pages": actual_page_count,
                "reliable_pages": len(reliable_page_numbers),
                "reliable_page_numbers": reliable_page_numbers,
                "inspection_coverage": inspection_coverage,
            }

            if inspection_issues:
                reliable_page_numbers, inspection_coverage = self._coverage(
                chunks,
                actual_page_count,
                inspection_issues,
                inspection_warnings,
            )
            self._remove_output(output_path)
            return {
                "status": "BLOCKED",
                "output_pdf": None,
                "redaction_count": 0,
                "unmapped_findings": [],
                "verification_passed": False,
                "inspection_issues": inspection_issues,
                "inspection_warnings": inspection_warnings,
                "total_pages": actual_page_count,
                "reliable_pages": len(reliable_page_numbers),
                "reliable_page_numbers": reliable_page_numbers,
                "inspection_coverage": inspection_coverage,
            }

        if self.pii_pipeline is None:
            self.pii_pipeline = PIIPipeline()
        redaction_count = 0
        verification_passed = False
        rescan_failed = False

        for attempt in range(2):
            try:
                self._remove_output(output_path)

                redaction_count = self.redactor.redact(
                    input_pdf=input_pdf,
                    output_pdf=output_pdf,
                    chunks=chunks,
                    pii_results=pii_results,
                )

                rescanned_chunks = (
                    self.extractor(output_path)
                    if attempt == 0
                    else self._extract_with_retry(output_path)
                )
                rescanned_pages = [chunk.page_number for chunk in rescanned_chunks]

                if (
                    len(rescanned_pages) != actual_page_count
                    or set(rescanned_pages) != expected_pages
                    or len(set(rescanned_pages)) != len(rescanned_pages)
                ):
                    raise ValueError("Fresh extraction did not cover every output page")

                rescan_issues, rescan_warnings = self._inspect_chunks(rescanned_chunks)

                rescanned_results = self.pii_pipeline.process_document_chunks(
                    rescanned_chunks
                )

                verification_passed = self.verifier.verify(
                    rescanned_results,
                    expected_pages=actual_page_count,
                )

                if not rescan_issues and not rescan_warnings and verification_passed:
                    inspection_issues = []
                    inspection_warnings = list(original_inspection_warnings)
                    rescan_failed = False
                    break

                inspection_issues = rescan_issues
                inspection_warnings = rescan_warnings
                rescan_failed = True

            except (OSError, ValueError, RuntimeError, TypeError) as exc:
                inspection_issues = [
                    {
                        "reason": "Redaction or rescan attempt failed",
                        "attempt": attempt + 1,
                        "error": str(exc),
                    }
                ]
                rescan_failed = True
                verification_passed = False

            if attempt == 0:
                try:
                    retry_chunks = self._extract_with_retry(input_path)
                    retry_pages = [chunk.page_number for chunk in retry_chunks]

                    if (
                        len(retry_pages) != actual_page_count
                        or set(retry_pages) != expected_pages
                        or len(set(retry_pages)) != len(retry_pages)
                    ):
                        raise ValueError(
                            "Retry extraction did not cover every input page"
                        )

                    chunks = retry_chunks


                    if inspection_issues or inspection_warnings:
                        raise ValueError("Input inspection is incomplete after retry")

                    pii_results = self.pii_pipeline.process_document_chunks(chunks)

                    unmapped_findings = find_unmapped(chunks, pii_results)
                    if unmapped_findings:
                        raise ValueError("PII findings remain unmapped after retry")

                except (OSError, ValueError, RuntimeError, TypeError) as exc:
                    inspection_issues = [
                        {
                            "reason": "Retry before redaction failed",
                            "error": str(exc),
                        }
                    ]
                    break

        if rescan_failed or not verification_passed:
            self._remove_output(output_path)

        if inspection_issues or not verification_passed or rescan_failed:
            status = "BLOCKED"
        elif inspection_warnings:
            status = "REVIEW_REQUIRED"
        else:
            status = "VERIFIED_NO_PII_FOUND"

        if status != "VERIFIED_NO_PII_FOUND":
            self._remove_output(output_path)

        reliable_page_numbers, inspection_coverage = self._coverage(
            chunks,
            actual_page_count,
            inspection_issues,
            inspection_warnings,
        )

        return {
            "status": status,
            "output_pdf": (
                str(output_path)
                if status == "VERIFIED_NO_PII_FOUND"
                and output_path.is_file()
                and output_path.stat().st_size > 0
                else None
            ),
            "redaction_count": redaction_count,
            "unmapped_findings": unmapped_findings,
            "verification_passed": verification_passed,
            "inspection_issues": inspection_issues,
            "inspection_warnings": inspection_warnings,
            "total_pages": actual_page_count,
            "reliable_pages": len(reliable_page_numbers),
            "reliable_page_numbers": reliable_page_numbers,
            "inspection_coverage": inspection_coverage,
        }
