import logging

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

from config.settings import (
    IGNORED_ORGANIZATIONS,
    TARGET_ENTITIES,
    CorpEmailRecognizer,
    InitialNameRecognizer,
    RobustPhoneRecognizer,
)
from models.schemas import DetectedEntityDetail, ExtractionInput, PIIOutput
from pii.token_manager import TokenManager
from pii.verifier import VerificationGate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PIIPipeline")


class PIIPipeline:

    def __init__(self):
        # 1. Configure Presidio with en_core_web_sm & ignore non-PII spaCy labels
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            "ner_model_configuration": {
                "labels_to_ignore": [
                    "CARDINAL",
                    "ORDINAL",
                    "QUANTITY",
                    "WORK_OF_ART",
                    "PERCENT",
                    "MONEY",
                ]
            },
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()

        # 2. Register custom recognizers
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        registry.add_recognizer(InitialNameRecognizer)
        registry.add_recognizer(CorpEmailRecognizer)
        registry.add_recognizer(RobustPhoneRecognizer)

        # 3. Create single shared AnalyzerEngine
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

        results = self.analyzer.analyze(
            text=raw_text, entities=TARGET_ENTITIES, language="en"
        )
        results = self._remove_overlapping_spans(results)

        detected_details = []

        for res in results:
            original_val = raw_text[res.start : res.end].strip()
            entity_type = res.entity_type

            # Skip false-positive acronyms (e.g., RACI, SWOT) tagged as ORGANIZATION
            if (
                entity_type == "ORGANIZATION"
                and original_val.upper() in IGNORED_ORGANIZATIONS
            ):
                continue

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
        """Processes multi-page chunks reusing a single TokenManager and shared Presidio engine."""
        shared_token_manager = TokenManager()
        outputs = []

        for chunk in chunks:
            out = self.process_chunk(
                chunk, shared_token_manager=shared_token_manager
            )
            outputs.append(out)

        return outputs

    def _remove_overlapping_spans(self, results):
        sorted_res = sorted(results, key=lambda x: (x.start, -x.score))
        filtered = []
        last_end = -1

        for res in sorted_res:
            if res.start >= last_end:
                filtered.append(res)
                last_end = res.end

        return filtered