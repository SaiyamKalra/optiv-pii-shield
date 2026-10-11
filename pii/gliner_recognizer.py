import logging
from typing import Any

from gliner import GLiNER
from presidio_analyzer import EntityRecognizer, RecognizerResult

logger = logging.getLogger("PIIPipeline")


class GlinerRecognizer(EntityRecognizer):
    """Custom Presidio Recognizer that uses GLiNER for zero-shot entity detection."""

    def __init__(
        self,
        model_name: str = "urchade/gliner_medium-v2.1",
        threshold: float = 0.40,
    ):
        self.label_mapping = {
            "person": "PERSON",
            "person name": "PERSON",
            "full name": "PERSON",
            "individual name": "PERSON",
            "executive name": "PERSON",
            "organization": "ORGANIZATION",
            "location": "LOCATION",
        }

        self.model_name = model_name
        self.threshold = threshold
        self.labels = list(self.label_mapping.keys())
        self.model = None

        supported_entities = list(set(self.label_mapping.values()))

        super().__init__(
            supported_entities=supported_entities,
            name="GlinerRecognizer",
        )

    def load(self) -> None:
        """Load GLiNER model weights when initialized by Presidio."""
        if getattr(self, "model", None) is None:
            logger.info(f"Loading GLiNER model: {self.model_name}...")
            try:
                self.model = GLiNER.from_pretrained(self.model_name, load_onnx=True)
                logger.info("Loaded ONNX optimized GLiNER model.")
            except (RuntimeError, ValueError, TypeError, OSError):
                self.model = GLiNER.from_pretrained(self.model_name)

    def analyze(
        self,
        text: str,
        entities: list[str] | None = None,
        nlp_artifacts: Any | None = None,
    ) -> list[RecognizerResult]:
        if self.model is None:
            self.load()

        results = []
        if not text or not text.strip():
            return results

        try:
            predictions = self.model.predict_entities(
                text,
                labels=self.labels,
                threshold=self.threshold,
            )

            for pred in predictions:
                target_entity = self.label_mapping.get(pred["label"])

                if entities and target_entity not in entities:
                    continue

                start = int(pred["start"])
                end = int(pred["end"])

                # Trim trailing/leading punctuation from predicted span bounds
                span_text = text[start:end]
                while span_text and span_text[-1] in ".,;:!?()[]{}'\"":
                    end -= 1
                    span_text = text[start:end]

                while span_text and span_text[0] in ".,;:!?()[]{}'\"":
                    start += 1
                    span_text = text[start:end]

                if start >= end:
                    continue

                results.append(
                    RecognizerResult(
                        entity_type=target_entity,
                        start=start,
                        end=end,
                        score=float(pred["score"]),
                        recognition_metadata={"recognizer_name": self.name},
                    )
                )
        except (RuntimeError, ValueError, TypeError, KeyError) as err:
            logger.error(f"GLiNER prediction failed: {err}")

        return results