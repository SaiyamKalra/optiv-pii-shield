import argparse
import json
import logging
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Restrict PyTorch thread thrashing on CPU for faster inference execution
import torch

torch.set_num_threads(4)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from pydantic import TypeAdapter
from sqlalchemy.exc import SQLAlchemyError

from config.settings import (
    MIN_NER_SCORE,
    MIN_PATTERN_SCORE,
    NER_RECOGNIZER_PREFIXES,
    TARGET_ENTITIES,
    ContextualNameRecognizer,
    CorpCreditCardRecognizer,
    CorpEmailRecognizer,
    DateOfBirthRecognizer,
    DirectorIdRecognizer,
    EmployeeIdRecognizer,
    NationalIdRecognizer,
    PanRecognizer,
    PassportRecognizer,
    RobustPhoneRecognizer,
    StreetAddressRecognizer,
    StrictSsnRecognizer,
    VendorIdRecognizer,
)
from database.db import init_db
from database.repository import save_pipeline_output
from models.schemas import DetectedEntityDetail, ExtractionInput, PIIOutput
from pii.person_filter import (
    DocumentVocabulary,
    canonical_names,
    is_person_name,
    is_single_name_word,
    name_tokens,
    repeated_first_names,
    surname_keys,
    surname_spans,
    trim_person_span,
)
from pii.token_manager import TokenManager
from pii.verifier import VerificationGate
from redaction.token_redaction.token_optimizer import TokenOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PIIPipeline")

# Entity priority matrix: structured IDs beat generic ML matches
ENTITY_PRIORITY = {
    "EMPLOYEE_ID": 0,
    "DIRECTOR_ID": 0,
    "VENDOR_ID": 0,
    "PASSPORT": 0,
    "PAN": 0,
    "NATIONAL_ID": 0,
    "EXCEPTION_ID": 0,
    "DOCUMENT_REFERENCE": 0,
    "CREDIT_CARD": 1,
    "US_SSN": 1,
    "IP_ADDRESS": 1,
    "EMAIL_ADDRESS": 1,
    "DATE_OF_BIRTH": 1,
    "STREET_ADDRESS": 1,
    "PHONE_NUMBER": 2,
    "PERSON": 3,
    "ORGANIZATION": 4,
    "LOCATION": 4,
}

PROPAGATED_NAME_SCORE = 0.9

DEFAULT_ENTITY_PRIORITY = 99


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    entity_type: str
    score: float
    recognizer: str = ""


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) bounds so span does not begin/end with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _recognizer_name(result) -> str:
    """Extract recognizer name from Presidio result metadata."""
    meta = getattr(result, "recognition_metadata", None) or {}
    return str(meta.get("recognizer_name", ""))


def _is_model_backed(span: _Span) -> bool:
    """True if spaCy's statistical model produced the span (not one of our regexes)."""
    return span.recognizer.casefold().startswith(
        tuple(name.casefold() for name in NER_RECOGNIZER_PREFIXES)
    )


def _min_score_for(span: _Span) -> float:
    """Determine confidence threshold based on recognizer origin."""
    recognizer_name = span.recognizer.casefold()

    ner_prefixes = tuple(name.casefold() for name in NER_RECOGNIZER_PREFIXES)

    if recognizer_name.startswith(ner_prefixes):
        return MIN_NER_SCORE

    return MIN_PATTERN_SCORE


class PIIPipeline:

    def __init__(self):
        try:
            init_db()
        except SQLAlchemyError as e:
            logger.warning(f"Failed to initialize database: {e}")

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": "en_core_web_lg",
                    "ner_model_configuration": {
                        "labels_to_ignore": [
                            "CARDINAL",
                            "ORDINAL",
                            "QUANTITY",
                            "WORK_OF_ART",
                            "PERCENT",
                            "MONEY",
                            "FAC",
                            "LAW",
                            "EVENT",
                            "PRODUCT",
                            # Organisations and bare places are not redacted.
                            "ORG",
                            "GPE",
                            "LOC",
                            "NORP",
                        ]
                    },
                }
            ],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en"])

        registry.remove_recognizer("PhoneRecognizer")

        registry.add_recognizer(CorpEmailRecognizer)
        registry.add_recognizer(RobustPhoneRecognizer)
        registry.add_recognizer(StrictSsnRecognizer)
        registry.add_recognizer(CorpCreditCardRecognizer)
        registry.add_recognizer(EmployeeIdRecognizer)
        registry.add_recognizer(DirectorIdRecognizer)
        registry.add_recognizer(VendorIdRecognizer)
        registry.add_recognizer(PassportRecognizer)
        registry.add_recognizer(PanRecognizer)
        registry.add_recognizer(NationalIdRecognizer)
        registry.add_recognizer(DateOfBirthRecognizer)
        registry.add_recognizer(StreetAddressRecognizer)
        registry.add_recognizer(ContextualNameRecognizer())
        # Policy numbers (GRP-POL-...) and exception IDs (EXC-...) are not personal
        # data, so DocumentReferenceRecognizer / ExceptionIdRecognizer are not registered.

        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, registry=registry
        )
        self.verifier = VerificationGate(analyzer_engine=self.analyzer)

    def process_chunk(
        self,
        payload: ExtractionInput,
        shared_token_manager: TokenManager = None,
    ) -> PIIOutput:
        vocab = DocumentVocabulary([payload.raw_text])
        spans = self._detect_spans(payload, vocab)
        spans = self._resolve_names([payload], {payload.page_number: spans}, vocab)
        return self._build_output(payload, spans[payload.page_number], shared_token_manager)

    def process_document_chunks(
        self, chunks: list[ExtractionInput]
    ) -> list[PIIOutput]:
        """Detect on every page first, then share what was learned across pages:
        a person's full name found on one page is redacted wherever it appears
        (tables often lack the context that made it detectable elsewhere)."""
        shared_token_manager = TokenManager()
        # What the document says about its own words: ordinary words (seen in
        # lowercase) and name parts confirmed by its email addresses.
        vocab = DocumentVocabulary(chunk.raw_text for chunk in chunks)
        spans_by_page = {chunk.page_number: self._detect_spans(chunk, vocab) for chunk in chunks}
        spans_by_page = self._resolve_names(chunks, spans_by_page, vocab)
        return [
            self._build_output(chunk, spans_by_page[chunk.page_number], shared_token_manager)
            for chunk in chunks
        ]

    def _detect_spans(self, payload: ExtractionInput, vocab: DocumentVocabulary) -> list[_Span]:
        """Candidate spans on one page: scored, filtered and (for PERSON) trimmed.
        Overlaps are not resolved yet."""
        raw_text = payload.raw_text
        if not raw_text or not raw_text.strip():
            return []

        results = self.analyzer.analyze(
            text=raw_text, entities=TARGET_ENTITIES, language="en"
        )

        candidates: list[_Span] = []

        for res in results:
            start, end = _trim_span(raw_text, res.start, res.end)
            if res.entity_type == "PERSON":
                start, end = trim_person_span(raw_text, start, end, vocab)
            if start >= end:
                continue

            candidates.append(
                _Span(
                    start=start,
                    end=end,
                    entity_type=res.entity_type,
                    score=res.score,
                    recognizer=_recognizer_name(res),
                )
            )

        valid_candidates: list[_Span] = []
        for span in candidates:
            if span.score < _min_score_for(span):
                continue

            val = raw_text[span.start : span.end]

            # A PERSON must be name-shaped with no ordinary words. Whether a span found
            # only by our regex pattern has enough evidence, and whether a single word
            # is kept, is decided per document in _resolve_names.
            if span.entity_type == "PERSON" and not (
                is_person_name(val, vocab) or is_single_name_word(val, vocab)
            ):
                continue

            valid_candidates.append(span)

        return valid_candidates

    def _resolve_names(
        self,
        chunks: list[ExtractionInput],
        spans_by_page: dict[int, list[_Span]],
        vocab: DocumentVocabulary,
    ) -> dict[int, list[_Span]]:
        """Document-level PERSON decisions, then overlap removal per page.

        - full names: those spaCy's model found, plus names found only by our regex
          pattern that have evidence (a part confirmed by an email address, or a word
          from a name the model found);
        - single words ("Raghavan") are kept only if they belong to such a name or are
          confirmed by an email address;
        - every other occurrence of an accepted full name is added as a PERSON span.
        """
        texts = {chunk.page_number: chunk.raw_text or "" for chunk in chunks}

        def value_of(page, span):
            return " ".join(texts[page][span.start : span.end].split())

        person_spans = [
            (page, span) for page, spans in spans_by_page.items()
            for span in spans if span.entity_type == "PERSON"
        ]
        model_names = {
            value_of(page, span) for page, span in person_spans
            if _is_model_backed(span) and is_person_name(value_of(page, span), vocab)
        }
        model_words = {t for name in model_names for t in name_tokens(name) if len(t) > 2}
        full_names = model_names | {
            value_of(page, span) for page, span in person_spans
            if not _is_model_backed(span)
            and is_person_name(value_of(page, span), vocab, model_backed=False, known_words=model_words)
        }
        accepted_names = full_names
        full_names = canonical_names(full_names)
        known_words = {t for name in full_names for t in name_tokens(name) if len(t) > 2}
        first_names = {
            words[0] for name in full_names
            if len(words := name_tokens(name)) > 1 and not words[0].endswith(".")
        }
        name_patterns = [
            re.compile(r"(?<![\w-])" + r"\s+".join(map(re.escape, name.split())) + r"(?![\w-])")
            for name in sorted(full_names, key=len, reverse=True)
        ]

        def keep(page, span):
            if span.entity_type != "PERSON":
                return True
            value = value_of(page, span)
            if value in accepted_names:
                return True  # a wider form is replaced by its canonical name below
            tokens = name_tokens(value)
            return len(tokens) == 1 and (
                tokens[0] in known_words or vocab.is_email_confirmed(tokens[0])
            )

        keys = surname_keys(full_names, vocab)
        pairs = repeated_first_names(texts.values(), keys, vocab)

        resolved: dict[int, list[_Span]] = {}
        for page, spans in spans_by_page.items():
            text = texts[page]
            kept = [span for span in spans if keep(page, span)]
            # Full names found anywhere, longest first; then every known surname with
            # its initials or first name ("T. Lindqvist", "Ifeoma C. Adeyemi").
            name_ranges = [m.span() for pattern in name_patterns for m in pattern.finditer(text)]
            name_ranges += surname_spans(text, keys, first_names, vocab, pairs)
            for a, b in name_ranges:
                # A detected identifier (email, ID...) is never overwritten.
                if any(s.entity_type != "PERSON" and s.start < b and s.end > a for s in kept):
                    continue
                # Overlapping names merge into one span: two names side by side
                # ("Devansh R. Kulkarni M. T. Achebe") both stay fully covered, and a
                # partial span ("Nathaniel", "Anonymous Helena") is replaced.
                for s in kept:
                    if s.recognizer == "NamePropagation" and s.start < b and s.end > a:
                        a, b = min(a, s.start), max(b, s.end)
                kept = [
                    s for s in kept
                    if not (s.entity_type == "PERSON" and s.start < b and s.end > a)
                ]
                kept.append(
                    _Span(
                        start=a,
                        end=b,
                        entity_type="PERSON",
                        score=PROPAGATED_NAME_SCORE,
                        recognizer="NamePropagation",
                    )
                )
            resolved[page] = self._remove_overlapping_spans(kept)
        return resolved

    def _build_output(
        self,
        payload: ExtractionInput,
        spans: list[_Span],
        shared_token_manager: TokenManager = None,
    ) -> PIIOutput:
        """Tokenise the page, run the verification gate and save the result."""
        raw_text = payload.raw_text
        file_id = payload.file_id
        page_num = payload.page_number

        if not raw_text or not raw_text.strip():
            result = PIIOutput(
                file_id=file_id,
                file_name=payload.file_name,
                source_url=payload.source_url,
                user_id=payload.user_id,
                page_number=page_num,
                status="FLAGGED_FOR_REVIEW",
                sanitized_text="",
                mapping_ledger={},
                detected_entities=[],
                verification_passed=True,
                detection_counts={},
            )

            try:
                save_pipeline_output(result)
            except SQLAlchemyError as e:
                logger.warning(f"Failed to persist output to DB: {e}")
            return result

        token_mgr = (
            shared_token_manager
            if shared_token_manager is not None
            else TokenManager()
        )

        detected_details = []
        for span in spans:
            original_val = raw_text[span.start : span.end]
            token = token_mgr.get_or_create_token(original_val, span.entity_type)

            detected_details.append(
                DetectedEntityDetail(
                    token=token,
                    entity_type=span.entity_type,
                    original_value=original_val,
                    start=span.start,
                    end=span.end,
                    score=span.score,
                    page_number=page_num,
                )
            )

        sanitized_text = raw_text
        for detail in sorted(
            detected_details, key=lambda x: x.start, reverse=True
        ):
            sanitized_text = (
                sanitized_text[: detail.start]
                + detail.token
                + sanitized_text[detail.end :]
            )

        verification_passed = self.verifier.verify(sanitized_text)

        result = PIIOutput(
            file_id=file_id,
            file_name=payload.file_name,
            source_url=payload.source_url,
            user_id=payload.user_id,
            page_number=page_num,
            status="SUCCESS" if verification_passed else "FLAGGED_FOR_REVIEW",
            sanitized_text=sanitized_text,
            mapping_ledger=token_mgr.mapping_ledger,
            detected_entities=detected_details,
            verification_passed=verification_passed,
            detection_counts=token_mgr.entity_counters,
        )

        try:
            save_pipeline_output(result)
        except SQLAlchemyError as e:
            logger.warning(f"Failed to persist output to DB: {e}")

        return result

    def _remove_overlapping_spans(self, results: list[_Span]) -> list[_Span]:
        """Keep one span per overlapping cluster."""
        sorted_res = sorted(
            results,
            key=lambda x: (
                ENTITY_PRIORITY.get(x.entity_type, DEFAULT_ENTITY_PRIORITY),
                -x.score,
                -(x.end - x.start),
            ),
        )

        filtered: list[_Span] = []
        for span in sorted_res:
            if not any(
                span.start < keep.end and span.end > keep.start for keep in filtered
            ):
                filtered.append(span)

        return sorted(filtered, key=lambda x: x.start)


def main():
    """Run pipeline on ExtractionInput JSON saved by an extractor."""
    parser = argparse.ArgumentParser(
        description="Run PII detection on extracted page chunks."
    )
    parser.add_argument(
        "chunks",
        help="ExtractionInput JSON file from an extractor's --save-chunks",
    )
    parser.add_argument(
        "--pages", help="Comma-separated page numbers to show, e.g. 7,22"
    )
    parser.add_argument(
        "--show-sanitized",
        action="store_true",
        help="Print each page's sanitized (tokenized) text",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="Save raw PIIOutput objects as JSON",
    )
    parser.add_argument(
        "--export-md",
        metavar="FILE",
        help="Save token-optimized, LLM-ready Markdown document (.md)",
    )
    args = parser.parse_args()

    for name in ("presidio-analyzer", "PIIPipeline"):
        logging.getLogger(name).setLevel(logging.ERROR)

    raw = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    chunks = TypeAdapter(list[ExtractionInput]).validate_python(raw)
    selected = {int(p) for p in args.pages.split(",")} if args.pages else None

    results = PIIPipeline().process_document_chunks(chunks)

    print(f"{chunks[0].file_name if chunks else args.chunks}: {len(results)} pages")
    print(f"{'page':>4}  {'status':<18} {'findings':>8}  types")
    totals = Counter()
    for r in results:
        types = Counter(e.entity_type for e in r.detected_entities)
        totals.update(types)
        if selected and r.page_number not in selected:
            continue
        summary = ", ".join(f"{t} {n}" for t, n in types.most_common())
        print(
            f"{r.page_number:>4}  {r.status:<18} {len(r.detected_entities):>8}  {summary}"
        )
        if args.show_sanitized:
            print(r.sanitized_text)
            print("-" * 80)

    flagged = [r.page_number for r in results if not r.verification_passed]
    unique_tokens = len(results[-1].mapping_ledger) if results else 0

    print(
        f"\nTotal findings: {sum(totals.values())} "
        f"({', '.join(f'{t} {n}' for t, n in totals.most_common())})"
    )
    print(f"Unique tokens: {unique_tokens}")
    print(f"Pages flagged by verifier: {flagged or 'none'}")

    if args.export_md:
        raw_full_text = "\n".join([c.raw_text for c in chunks if c.raw_text])
        optimizer = TokenOptimizer()
        optimized_md = optimizer.optimize_output(
            pii_outputs=results, raw_original_text=raw_full_text, chunks=chunks
        )
        metrics = optimizer.calculate_metrics(raw_full_text, optimized_md)

        out_path = Path(args.export_md)
        out_path.write_text(optimized_md, encoding="utf-8")
        print(
            f"\nSaved Token-Optimized Markdown to {out_path.resolve()}"
            f"\nEstimated Tokens Saved: {metrics['tokens_saved']} ({metrics['savings_percentage']}% reduction)"
        )

    if args.json:
        out = [
            r.model_dump(mode="json")
            for r in results
            if not selected or r.page_number in selected
        ]
        Path(args.json).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"Saved raw PIIOutput for {len(out)} page(s) to {Path(args.json).resolve()}"
        )


if __name__ == "__main__":
    main()