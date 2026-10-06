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
from models.schemas import DetectedEntityDetail, ExtractionInput, PIIOutput
from pii.token_manager import TokenManager
from pii.verifier import VerificationGate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PIIPipeline")


class PIIPipeline:

    def __init__(self):
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

        # CRITICAL: Remove Presidio's built-in default phone recognizer to stop IP/CC slicing
        registry.remove_recognizer("PhoneRecognizer")

        # Add our clean, bounded recognizers
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
            return PIIOutput(
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

        token_mgr = (
            shared_token_manager
            if shared_token_manager is not None
            else TokenManager()
        )

        # 1. Analyze text with Presidio
        results = self.analyzer.analyze(
            text=raw_text, entities=TARGET_ENTITIES, language="en"
        )

        # 2. Filter predictions: drop low confidence (< 0.60) & blacklisted words
        filtered_results = []
        for res in results:
            val = raw_text[res.start : res.end].strip()

            if res.score < 0.60:
                continue

            # Drop false-positive headers/acronyms
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
            entity_type = res.entity_type

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

        # 4. Replace tokens right-to-left
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

        return PIIOutput(
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