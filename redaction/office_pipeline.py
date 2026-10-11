from pathlib import Path

from extraction.docx_extractor import extract_docx
from extraction.pptx_extractor import extract_pptx
from pii.pipeline import PIIPipeline
from redaction.docx_redactor import DOCXRedactor
from redaction.pptx_redactor import PPTXRedactor


class OfficeRedactionPipeline:
    def __init__(self, pii_pipeline=None):
        self.pii_pipeline = pii_pipeline or PIIPipeline()

    @staticmethod
    def _scan_chunks(chunks):
        """Include hidden XML content in the PII scan."""
        scanned = []

        for chunk in chunks:
            hidden = getattr(chunk, "hidden_content", None) or []
            hidden_text = "\n".join(
                str(getattr(item, "text", "") or "") for item in hidden
            )

            # Scan visible and hidden text together.
            updated = chunk.model_copy(
                update={"raw_text": ((chunk.raw_text or "") + "\n" + hidden_text)}
            )
            scanned.append(updated)

        return scanned

    @staticmethod
    def _unsafe_chunks(chunks):
        issues = []

        if not chunks:
            return [{"reason": "No pages or slides were extracted."}]

        page_numbers = [chunk.page_number for chunk in chunks]

        if len(page_numbers) != len(set(page_numbers)):
            issues.append({"reason": "Duplicate page or slide numbers."})

        if sorted(page_numbers) != list(range(1, len(chunks) + 1)):
            issues.append({"reason": "Page or slide coverage is incomplete."})

        for chunk in chunks:
            page = chunk.page_number

            if getattr(chunk, "extraction_errors", None):
                issues.append(
                    {
                        "page_number": page,
                        "reason": "Extraction errors prevent complete inspection.",
                    }
                )

            if getattr(chunk, "unread_regions", None):
                issues.append(
                    {
                        "page_number": page,
                        "reason": "Unread regions cannot be safely redacted.",
                    }
                )

            if getattr(chunk, "images_skipped_count", 0):
                issues.append(
                    {
                        "page_number": page,
                        "reason": "Skipped images may contain PII.",
                    }
                )

            if getattr(chunk, "low_confidence_word_count", 0):
                issues.append(
                    {
                        "page_number": page,
                        "reason": "Low-confidence OCR prevents safe certification.",
                    }
                )

            if (
                getattr(chunk, "ocr_used", False)
                or getattr(chunk, "extraction_method", "") in ("ocr", "hybrid")
            ) and getattr(chunk, "ocr_mean_confidence", None) is None:
                issues.append(
                    {
                        "page_number": page,
                        "reason": "OCR confidence is unavailable.",
                    }
                )

            if getattr(chunk, "images_ocr_count", 0):
                issues.append(
                    {
                        "page_number": page,
                        "reason": "Image redaction is not implemented.",
                    }
                )

            for box in getattr(chunk, "bounding_boxes", None) or []:
                if getattr(box, "coord_space", None) == "image_px":
                    issues.append(
                        {
                            "page_number": page,
                            "reason": "Image-coordinate PII cannot be safely redacted.",
                        }
                    )
                    break

        return issues

    def process(self, input_path, output_path):
        source = Path(input_path)
        destination = Path(output_path)

        if source.resolve() == destination.resolve():
            raise ValueError("Output must differ from the original file.")

        if not source.is_file():
            raise FileNotFoundError(source)

        extension = source.suffix.lower()

        if extension == ".docx":
            extractor = extract_docx
            redactor = DOCXRedactor()
        elif extension == ".pptx":
            extractor = extract_pptx
            redactor = PPTXRedactor()
        else:
            raise ValueError("Only DOCX and PPTX are supported.")

        destination.unlink(missing_ok=True)

        try:
            original_chunks = extractor(str(source))
            if not original_chunks:
                raise ValueError("Document extraction returned no content.")

            issues = self._unsafe_chunks(original_chunks)
            if issues:
                return {
                    "status": "BLOCKED",
                    "output_path": None,
                    "verification_passed": False,
                    "inspection_issues": issues,
                }

            scan_chunks = self._scan_chunks(original_chunks)
            findings = self.pii_pipeline.process_document_chunks(scan_chunks)

            if not any(result.detected_entities for result in findings):
                return {
                    "status": "BLOCKED",
                    "output_path": None,
                    "verification_passed": False,
                    "inspection_issues": [
                        {
                            "reason": (
                                "No PII was detected; this implementation "
                                "does not certify Office files as safe."
                            )
                        }
                    ],
                }

            (
                redactor.redact(
                    input_docx=str(source),
                    output_docx=str(destination),
                    chunks=original_chunks,
                    pii_results=findings,
                )
                if extension == ".docx"
                else redactor.redact(
                    input_pptx=str(source),
                    output_pptx=str(destination),
                    chunks=original_chunks,
                    pii_results=findings,
                )
            )

            # Fresh extraction and independent rescan of the saved file.
            rescanned_chunks = extractor(str(destination))

            if len(rescanned_chunks) != len(original_chunks):
                raise ValueError(
                    "Output extraction did not cover the original pages/slides."
                )

            issues = self._unsafe_chunks(rescanned_chunks)
            if issues:
                raise ValueError(f"Output contains unverified content: {issues}")

            rescanned_text = self._scan_chunks(rescanned_chunks)
            remaining = self.pii_pipeline.process_document_chunks(rescanned_text)

            if any(result.detected_entities for result in remaining):
                raise ValueError("PII remains in the saved Office document.")

            if not destination.is_file() or destination.stat().st_size == 0:
                raise ValueError("Redacted output file is missing or empty.")

            return {
                "status": "VERIFIED_NO_PII_FOUND",
                "output_path": str(destination),
                "verification_passed": True,
                "inspection_issues": [],
                "findings": findings,
                "total_pages": len(original_chunks),
                "reliable_pages": len(original_chunks),
                "reliable_page_numbers": [
                    chunk.page_number for chunk in original_chunks
                ],
                "inspection_coverage": 100.0,
            }

        except Exception as exc:
            destination.unlink(missing_ok=True)
            return {
                "status": "BLOCKED",
                "output_path": None,
                "verification_passed": False,
                "inspection_issues": [
                    {
                        "reason": "Office redaction or verification failed",
                        "error": str(exc),
                    }
                ],
            }
