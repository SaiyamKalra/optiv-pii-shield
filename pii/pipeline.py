import logging
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from sqlalchemy.exc import SQLAlchemyError

from config.settings import (
    IGNORED_WORDS,
    MIN_NER_SCORE,
    MIN_PATTERN_SCORE,
    NER_RECOGNIZER_PREFIXES,
    TARGET_ENTITIES,
    CorpCreditCardRecognizer,
    CorpEmailRecognizer,
    DirectorIdRecognizer,
    DocumentReferenceRecognizer,
    EmployeeIdRecognizer,
    ExceptionIdRecognizer,
    RobustPhoneRecognizer,
    StrictSsnRecognizer,
)
from database.db import init_db
from database.repository import save_pipeline_output
from models.schemas import DetectedEntityDetail, ExtractionInput, PIIOutput
from pii.token_manager import TokenManager
from pii.verifier import VerificationGate
from redaction.token_redaction.token_optimizer import TokenOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PIIPipeline")


class PIIPipeline:

    def __init__(self):
        try:
            init_db()
        except SQLAlchemyError as e:
            logger.warning(f"Failed to initialize database: {e}")

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_lg"}],
            "ner_model_configuration": {
                "labels_to_ignore": [
                    "CARDINAL",
                    "ORDINAL",
                    "QUANTITY",
                    "WORK_OF_ART",
                    "PERCENT",
                    "MONEY",
                    "FAC",
                ]
            },
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()

        registry.remove_recognizer("PhoneRecognizer")

        # registry.add_recognizer(InitialNameRecognizer)
        registry.add_recognizer(CorpEmailRecognizer)
        registry.add_recognizer(RobustPhoneRecognizer)
        registry.add_recognizer(StrictSsnRecognizer)
        registry.add_recognizer(CorpCreditCardRecognizer)
        registry.add_recognizer(EmployeeIdRecognizer)
        registry.add_recognizer(DirectorIdRecognizer)
        registry.add_recognizer(ExceptionIdRecognizer)
        registry.add_recognizer(DocumentReferenceRecognizer)
        # Registration order does not decide overlaps: ENTITY_PRIORITY does
        # (EMPLOYEE_ID beats PHONE_NUMBER on the same digits).
        # registry.add_recognizer(EmployeeIDRecognizer)

        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, registry=registry
        )
        self.verifier = VerificationGate(analyzer_engine=self.analyzer)

    def process_chunk(
        self,
        payload: ExtractionInput,
        shared_token_manager: TokenManager = None,
    ) -> PIIOutput:
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

        results = self.analyzer.analyze(
            text=raw_text, entities=TARGET_ENTITIES, language="en"
        )
        for res in results:
            print(
                f"DETECTED: {raw_text[res.start:res.end]!r} "
                f"| ENTITY: {res.entity_type} "
                f"| SCORE: {res.score:.2f} "
                f"| RECOGNIZER: {_recognizer_name(res)}"
            )
        candidates: list[_Span] = []
        for res in results:
            # Trim whitespace out of the span itself, so the offsets, the token
            # and the recorded original value always describe the same text.
            start, end = _trim_span(raw_text, res.start, res.end)
            if start >= end:
                continue

            span = _Span(
                start=start,
                end=end,
                entity_type=res.entity_type,
                score=res.score,
                recognizer=_recognizer_name(res),
            )

            if span.score < _min_score_for(span):
                continue

            val = raw_text[start:end]
            if val.upper() in IGNORED_WORDS or any(
                w in val.upper()
                for w in [
                    "RACI MATRIX",
                    "COMPLIANCE SIGN-OFF",
                    "INFRASTRUCTURE NODES",
                    "ROSTER APPENDIX",
                    "SLA & KPI",
                ]
            ):
                continue

            candidates.append(span)

        # Deduplicate overlapping spans: structured identifiers beat generic
        # digit-heavy matches (see _remove_overlapping_spans).
        results = self._remove_overlapping_spans(candidates)

        detected_details = []
        for span in results:
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

    def process_document_chunks(
        self, chunks: list[ExtractionInput]
    ) -> list[PIIOutput]:
        shared_token_manager = TokenManager()
        outputs = []
        for chunk in chunks:
            out = self.process_chunk(
                chunk, shared_token_manager=shared_token_manager
            )
            outputs.append(out)
        return outputs

    def _remove_overlapping_spans(self, results: list["_Span"]) -> list["_Span"]:
        """Keep one span per overlapping cluster.

        Ordering is strictly by entity priority first (lower number wins), then
        by recognizer score, then by span length. The priority rule is what
        makes a contextual structured identifier (EMPLOYEE_ID, priority 0) beat
        any generic multi-digit match such as PHONE_NUMBER (priority 2) that
        fires on the same digits. Score only breaks ties within a priority tier,
        so a low-confidence structured ID can never lose to a generic regex.
        """
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


# ── Span helpers ──────────────────────────────────────────────────────────────

# Lower number = higher priority. EMPLOYEE_ID must win over the generic phone /
# SSN / card regexes that also fire on its digits.
ENTITY_PRIORITY = {
    "EMPLOYEE_ID": 0,
    "DIRECTOR_ID": 0,
    "EXCEPTION_ID": 0,
    "DOCUMENT_REFERENCE": 0,
    "CREDIT_CARD": 1,
    "US_SSN": 1,
    "IP_ADDRESS": 1,
    "EMAIL_ADDRESS": 1,
    "PHONE_NUMBER": 2,
    "PERSON": 3,
    "ORGANIZATION": 4,
    "LOCATION": 4,
}

DEFAULT_ENTITY_PRIORITY = 99


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    entity_type: str
    score: float
    recognizer: str = ""


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink [start, end) so it does not begin or end with whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _recognizer_name(result) -> str:
    """Name of the Presidio recognizer that produced *result*, if reported."""
    meta = getattr(result, "recognition_metadata", None) or {}
    return str(meta.get("recognizer_name", ""))


def _min_score_for(span: _Span) -> float:
    """Confidence floor for a span, chosen by where it came from."""
    recognizer_name = span.recognizer.casefold()
    ner_prefixes = tuple(
        name.casefold() for name in NER_RECOGNIZER_PREFIXES
    )

    if recognizer_name.startswith(ner_prefixes):
        return MIN_NER_SCORE

    return MIN_PATTERN_SCORE


def main():
    """Run the pipeline on ExtractionInput JSON saved by an extractor (--save-chunks)."""
    import argparse
    import json
    from collections import Counter
    from pathlib import Path

    from pydantic import TypeAdapter

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
        help="Save the raw PIIOutput objects as JSON (includes original values, i.e. PII)",
    )
    parser.add_argument(
        "--export-md",
        metavar="FILE",
        help="Save the token-optimized, LLM-ready Markdown document (.md)",
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
    print(
        f"\nTotal findings: {sum(totals.values())} "
        f"({', '.join(f'{t} {n}' for t, n in totals.most_common())})"
    )
    print(f"Unique tokens: {len(results[-1].mapping_ledger) if results else 0}")
    print(f"Pages flagged by verifier: {flagged or 'none'}")

    # Process Token Optimizer downstream export if flag is provided
    if args.export_md:
        raw_full_text = "\n".join([c.raw_text for c in chunks if c.raw_text])
        optimizer = TokenOptimizer()
        optimized_md = optimizer.optimize_output(
            pii_outputs=results, raw_original_text=raw_full_text
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