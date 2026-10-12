import logging
import os
import re
from dataclasses import dataclass
from typing import ClassVar

try:
    import numpy as np
except ImportError:
    np = None

try:
    import joblib
except ImportError:
    joblib = None

logger = logging.getLogger(__name__)


@dataclass
class CandidateFeatureVector:
    base_score: float
    has_person_context: float
    has_id_context: float
    is_inside_table: float
    span_length: int
    digit_ratio: float
    is_title_case_name: float
    has_international_phone_prefix: float


class PIIRiskCalibrator:

    # RUF012 Fix: ClassVar annotation for mutable sets
    PERSON_TRIGGERS: ClassVar[set[str]] = {
        "name", "person", "contact", "individual", "author", "auditor", "sign-off",
        "officer", "lead", "head", "manager", "director", "executive", "deputy", 
        "chair", "owner", "delegate", "approver", "reviewer", "custodian",
        "representative", "coordinator", "analyst", "specialist", "sponsor",
        "local", "global", "regional", "risk", "compliance", "security",
        "relationship", "relationship owner", "cfo", "ciso", "cco", "ceo"
    }

    ID_TRIGGERS: ClassVar[set[str]] = {
        "pan", "ssn", "passport", "tax id", "ref", "reference", "id number", "code",
        "phone", "mobile", "tel", "fax", "contact number", "account", "credit", "card"
    }

    def __init__(self, model_path: str | None = None):
        self.model = None
        if model_path and os.path.exists(model_path) and joblib is not None:
            # BLE001 & S110 Fix: Logging exception instead of blind silent pass
            try:
                self.model = joblib.load(model_path)
            except (FileNotFoundError, ValueError, EOFError, OSError) as e:
                logger.debug(f"Could not load regressor model from {model_path}: {e}")

    def extract_features(
        self,
        text: str,
        start: int,
        end: int,
        base_score: float,
        column_header: str | None = None,
        is_inside_table: bool = False,
    ) -> CandidateFeatureVector:
        """Extracts contextual, syntactic, and tabular layout features around a candidate span."""
        span_text = text[start:end].strip()
        
        window_before = text[max(0, start - 40):start].lower()
        window_after = text[end:min(len(text), end + 40)].lower()
        combined_context = f"{window_before} {window_after}"

        if column_header:
            combined_context += f" {column_header.lower()}"

        has_person_ctx = float(any(kw in combined_context for kw in self.PERSON_TRIGGERS))
        has_id_ctx = float(any(kw in combined_context for kw in self.ID_TRIGGERS))

        span_len = len(span_text)
        digit_count = sum(c.isdigit() for c in span_text)
        digit_ratio = digit_count / max(1, span_len)

        is_title_name = float(
            bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-zA-Z\-]+)+\b", span_text))
        )

        has_phone_prefix = float(bool(re.search(r"^\+\d{1,3}\b", span_text)))

        return CandidateFeatureVector(
            base_score=base_score,
            has_person_context=has_person_ctx,
            has_id_context=has_id_ctx,
            is_inside_table=float(is_inside_table or bool(column_header)),
            span_length=span_len,
            digit_ratio=digit_ratio,
            is_title_case_name=is_title_name,
            has_international_phone_prefix=has_phone_prefix,
        )

    def predict_confidence(
        self,
        text: str,
        start: int,
        end: int,
        base_score: float,
        recognizer_name: str = "",
        column_header: str | None = None,
        is_inside_table: bool = False,
    ) -> float:
        """Computes calibrated probability score (0.0 - 1.0)."""
        feats = self.extract_features(
            text=text,
            start=start,
            end=end,
            base_score=base_score,
            column_header=column_header,
            is_inside_table=is_inside_table,
        )

        if self.model is not None and np is not None:
            X = np.array([[
                feats.base_score,
                feats.has_person_context,
                feats.has_id_context,
                feats.is_inside_table,
                feats.span_length,
                feats.digit_ratio,
                feats.is_title_case_name,
                feats.has_international_phone_prefix,
            ]])
            return float(self.model.predict_proba(X)[0][1])

        score = base_score

        if feats.is_title_case_name and (feats.has_person_context or feats.is_inside_table):
            score += 0.35 

        elif feats.has_international_phone_prefix or (feats.has_id_context and feats.digit_ratio > 0.35):
            score += 0.25

        elif not feats.has_person_context and not feats.has_id_context and feats.digit_ratio > 0.8:
            score -= 0.20

        return min(1.0, max(0.0, score))

    def harvest_table_cell_candidates(
        self, cell_text: str, column_header: str | None = None
    ) -> list[tuple[int, int, str, float]]:
        """Scans raw table cells for missed names or phone numbers regardless of header context."""
        extra_spans = []
        normalized_text = re.sub(r"\s+", " ", cell_text).strip()
        if not normalized_text:
            return extra_spans

        header_lower = (column_header or "").lower()
        is_person_col = any(h in header_lower for h in self.PERSON_TRIGGERS)

        name_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-zA-Z\-]+)+)\b")
        for match in name_pattern.finditer(cell_text):
            val = match.group(1).strip()
            if val.upper() not in {
                "LOCAL RISK", "RISK OFFICER", "LEGAL ENTITY", "RELATIONSHIP OWNER",
                "RISK COMMITTEE", "GROUP EXECUTIVE", "DATA PROTECTION"
            }:
                base_harvest_score = 0.55 if is_person_col else 0.45
                extra_spans.append((match.start(1), match.end(1), "PERSON", base_harvest_score))

        phone_pattern = re.compile(
            r"(?:\+\d{1,3}[\s\-\.]*)?(?:\(?\d{2,4}\)?[\s\-\.]*)*\d{3,4}[\s\-\.]*\d{3,4}"
        )
        for p in phone_pattern.finditer(cell_text):
            matched_str = p.group(0)
            if sum(c.isdigit() for c in matched_str) >= 7:
                extra_spans.append((p.start(), p.end(), "PHONE_NUMBER", 0.65))

        return extra_spans