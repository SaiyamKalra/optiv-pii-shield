# pii/pipeline.py

import logging

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

from config.settings import (
    IGNORED_WORDS,
    TARGET_ENTITIES,
    CorpCreditCardRecognizer,
    CorpEmailRecognizer,
    InitialNameRecognizer,
    RobustPhoneRecognizer,
    StrictSsnRecognizer,
)
from database.db import init_db
from database.repository import save_pipeline_output
from models.schemas import DetectedEntityDetail, ExtractionInput, PIIOutput
from pii.token_manager import TokenManager
from pii.verifier import VerificationGate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PIIPipeline")


class PIIPipeline:

    def __init__(self):
        try:
            init_db()
        except Exception as e:
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

        registry.add_recognizer(InitialNameRecognizer)
        registry.add_recognizer(CorpEmailRecognizer)
        registry.add_recognizer(RobustPhoneRecognizer)
        registry.add_recognizer(StrictSsnRecognizer)
        registry.add_recognizer(CorpCreditCardRecognizer)

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
                status="SUCCESS",
                sanitized_text="",
                mapping_ledger={},
                detected_entities=[],
                verification_passed=True,
                detection_counts={},
            )

            try:
                save_pipeline_output(result)
            except Exception as e:
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

        filtered_results = []
        for res in results:
            val = raw_text[res.start : res.end].strip()

            if res.score < 0.60:
                continue

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

            filtered_results.append(res)

        # 3. Deduplicate overlapping spans
        results = self._remove_overlapping_spans(filtered_results)

        detected_details = []
        for res in results:
            original_val = raw_text[res.start : res.end].strip()
            entity_type = res.entity_type  # <--- RESTORED THIS LINE

            token = token_mgr.get_or_create_token(original_val, entity_type)

            detected_details.append(
                DetectedEntityDetail(
                    token=token,
                    entity_type=entity_type,
                    original_value=original_val,
                    start=res.start,
                    end=res.end,
                    score=res.score,
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
        except Exception as e:
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

    def _remove_overlapping_spans(self, results):
        priority_map = {
            "CREDIT_CARD": 1,
            "US_SSN": 1,
            "IP_ADDRESS": 1,
            "EMAIL_ADDRESS": 1,
            "PHONE_NUMBER": 2,
            "PERSON": 3,
            "ORGANIZATION": 4,
            "LOCATION": 4,
        }

        sorted_res = sorted(
            results,
            key=lambda x: (
                priority_map.get(x.entity_type, 99),
                -x.score,
                -(x.end - x.start),
            ),
        )

        filtered = []
        for res in sorted_res:
            if not any(
                res.start < keep.end and res.end > keep.start for keep in filtered
            ):
                filtered.append(res)

        return sorted(filtered, key=lambda x: x.start)


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
    args = parser.parse_args()

    for name in ("presidio-analyzer", "PIIPipeline"):
        logging.getLogger(name).setLevel(logging.ERROR)

    raw = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    chunks = TypeAdapter(list[ExtractionInput]).validate_python(raw)
    selected = {int(p) for p in args.pages.split(",")} if args.pages else None

    # Always process every page so tokens stay consistent across the document.
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