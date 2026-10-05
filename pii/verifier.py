from presidio_analyzer import AnalyzerEngine

from config.settings import HIGH_RISK_VERIFICATION_ENTITIES


class VerificationGate:

    def __init__(self,analyzer_engine: AnalyzerEngine):
        self.analyzer = analyzer_engine

    def verify(self, sanitized_text: str) -> bool:

        if not sanitized_text.strip():
            return True

        secondary_results = self.analyzer.analyze(
            text=sanitized_text,
            entities=HIGH_RISK_VERIFICATION_ENTITIES,
            language="en",
            score_threshold=0.7,
        )

        return len(secondary_results) == 0