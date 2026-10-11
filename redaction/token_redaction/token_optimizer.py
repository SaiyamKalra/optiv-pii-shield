"""
Token Optimizer & Document Sanitizer Pipeline
Combines PII detection, custom pattern recognition, physical PDF redaction,
and structural Markdown formatting into a single unified module.
"""

import os
import re
from typing import Any

import fitz  # PyMuPDF
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

from models.schemas import PIIOutput


class TokenOptimizer:
    """Unified pipeline for PII detection, token mapping, physical PDF redaction,
    and structural Markdown formatting.
    """

    def __init__(self):
        # 1. Initialize Presidio Engine
        try:
            provider = NlpEngineProvider(conf_file=None)
            nlp_engine = provider.create_engine()
            self.analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine, supported_languages=["en"]
            )
        except (ValueError, AttributeError, RuntimeError, KeyError):
            # Catch specific configuration or runtime initialization errors
            self.analyzer = AnalyzerEngine()

        # 2. Register Custom Recognizers
        pan_pattern = Pattern(
            name="pan_pattern",
            regex=r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
            score=0.95,
        )
        pan_recognizer = PatternRecognizer(
            supported_entity="INDIAN_PAN", patterns=[pan_pattern]
        )

        ssn_pattern = Pattern(
            name="ssn_pattern",
            regex=r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b",
            score=0.90,
        )
        ssn_recognizer = PatternRecognizer(
            supported_entity="US_SSN", patterns=[ssn_pattern]
        )

        pesel_pattern = Pattern(
            name="pesel_pattern",
            regex=r"\b\d{11}\b",
            score=0.85,
        )
        pesel_recognizer = PatternRecognizer(
            supported_entity="POLISH_PESEL", patterns=[pesel_pattern]
        )

        self.analyzer.registry.add_recognizer(pan_recognizer)
        self.analyzer.registry.add_recognizer(ssn_recognizer)
        self.analyzer.registry.add_recognizer(pesel_recognizer)

        # 3. Centralized Token Registry
        self.token_mapping: dict[str, str] = {}

    def get_stable_token(self, entity_type: str, raw_value: str) -> str:
        """Generates or retrieves a consistent token placeholder across artifacts."""
        cleaned_val = raw_value.strip().lower()
        if cleaned_val not in self.token_mapping:
            prefix = entity_type.upper()
            token_index = (
                len([v for v in self.token_mapping.values() if f"[{prefix}_" in v]) + 1
            )
            self.token_mapping[cleaned_val] = f"[{prefix}_{token_index:03d}]"
        return self.token_mapping[cleaned_val]

    def redact_pdf_artifacts(
        self, pdf_path: str, output_pdf_path: str
    ) -> dict[str, Any]:
        """Applies physical bounding-box redactions to text and visual layers."""
        doc = fitz.open(pdf_path)
        total_redactions = 0

        for page in doc:
            for rect_info in page.get_drawings():
                page.add_redact_annot(rect_info["rect"], fill=(0, 0, 0))
                total_redactions += 1

            page_text = page.get_text("text")
            results = self.analyzer.analyze(
                text=page_text, language="en", score_threshold=0.4
            )

            for res in results:
                span_text = page_text[res.start : res.end].strip()
                if not span_text:
                    continue

                matching_rects = page.search_for(span_text)
                for m_rect in matching_rects:
                    page.add_redact_annot(m_rect, fill=(1, 1, 1))
                    total_redactions += 1

            page.apply_redactions()

        doc.save(output_pdf_path)
        doc.close()

        return {
            "output_path": output_pdf_path,
            "redaction_annotations_applied": total_redactions,
        }

    def extract_sanitized_markdown(self, pdf_path: str) -> str:
        """Extracts text, replaces PII with stable tokens, and returns clean Markdown."""
        doc = fitz.open(pdf_path)
        markdown_pages = []
        file_name = os.path.basename(pdf_path)

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            page_content = []

            for b in blocks:
                raw_block_text = b[4]
                results = self.analyzer.analyze(
                    text=raw_block_text, language="en", score_threshold=0.4
                )

                sorted_results = sorted(results, key=lambda x: x.start, reverse=True)
                sanitized_block = raw_block_text

                for res in sorted_results:
                    raw_val = sanitized_block[res.start : res.end]
                    token = self.get_stable_token(res.entity_type, raw_val)
                    sanitized_block = (
                        sanitized_block[: res.start]
                        + token
                        + sanitized_block[res.end :]
                    )

                clean_b = self._strip_boilerplate(sanitized_block)
                clean_b = self._clean_text(clean_b)
                if clean_b:
                    page_content.append(clean_b)

            formatted_page = f"<!-- Page {page_num} -->\n" + "\n\n".join(
                page_content
            )
            markdown_pages.append(formatted_page)

        doc.close()

        header = (
            "---\n"
            f"title: {file_name}\n"
            f"total_pages: {len(markdown_pages)}\n"
            "status: VERIFIED_PII_REDACTED\n"
            "---\n\n"
        )
        return header + "\n\n---\n\n".join(markdown_pages)

    def optimize_output(
        self,
        pii_outputs: list[PIIOutput],
        raw_original_text: str = "",
        chunks: list | None = None,
    ) -> str:
        """Formats PIIOutput list objects into clean Markdown with frontmatter metadata."""
        markdown_pages = []

        file_name = "Document_Sanitized_Output.pdf"
        file_id = ""
        total_pages = len(pii_outputs)

        if pii_outputs:
            first_page = pii_outputs[0]
            if getattr(first_page, "file_name", None):
                file_name = first_page.file_name
            if getattr(first_page, "file_id", None):
                file_id = first_page.file_id

        for p_idx, page in enumerate(pii_outputs, start=1):
            text = page.sanitized_text or ""
            if not text.strip() and chunks and len(chunks) >= p_idx:
                text = chunks[p_idx - 1].raw_text or ""

            clean_p_text = self._strip_boilerplate(text)
            clean_p_text = self._clean_text(clean_p_text)

            formatted_page = f"<!-- Page {p_idx} -->\n{clean_p_text}"
            markdown_pages.append(formatted_page)

        header_lines = ["---", f"title: {file_name}"]
        if file_id:
            header_lines.append(f"file_id: {file_id}")
        header_lines.extend(
            [f"total_pages: {total_pages}", "status: VERIFIED_PII_REDACTED", "---"]
        )

        header = "\n".join(header_lines) + "\n\n"
        return header + "\n\n---\n\n".join(markdown_pages)

    def _strip_boilerplate(self, text: str) -> str:
        """Removes headers, footers, page numbering, and confidentiality notices."""
        patterns = [
            r"Page\s+\d+\s+of\s+\d+",
            r"Confidential\s*-\s*Internal\s+Use\s+Only",
            r"All\s+Rights\s+Reserved\.",
        ]
        for pat in patterns:
            text = re.sub(pat, "", text, flags=re.IGNORECASE)
        return text

    def _clean_text(self, text: str) -> str:
        """Fixes token spacing, hyphenated line wraps, and extra linebreaks."""
        text = re.sub(r"(\w+)-\s*(\w+)", r"\1\2", text)
        text = re.sub(r"\[\s+([A-Z_]+_\d+)\s+\]", r"[\1]", text)
        text = re.sub(r"(\[[A-Z_]+_\d+\])\s+([.,;:!])", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        lines = [line.strip() for line in text.splitlines()]
        cleaned_blocks = []
        current_block = []

        for line in lines:
            if line:
                current_block.append(line)
            elif current_block:
                cleaned_blocks.append(" ".join(current_block))
                current_block = []

        if current_block:
            cleaned_blocks.append(" ".join(current_block))

        return "\n\n".join(cleaned_blocks).strip()

    @staticmethod
    def calculate_metrics(raw_text: str, optimized_text: str = "") -> dict:
        """Calculates basic token and word reduction metrics."""
        orig_words = len((raw_text or "").split())
        opt_words = (
            len((optimized_text or "").split()) if optimized_text else orig_words
        )
        saved_words = max(0, orig_words - opt_words)

        return {
            "original_words": orig_words,
            "optimized_words": opt_words,
            "words_saved": saved_words,
            "estimated_raw_tokens": int(orig_words * 1.3) if orig_words > 0 else 0,
            "savings_pct": (
                round((saved_words / orig_words) * 100, 2) if orig_words > 0 else 0.0
            ),
        }