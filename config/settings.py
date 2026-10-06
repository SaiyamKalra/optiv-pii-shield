from presidio_analyzer import Pattern, PatternRecognizer

INITIAL_NAME_PATTERN = Pattern(
    name="initial_name_pattern",
    regex=(
        r"\b[A-Z][a-z]+(?:-[A-Z][a-z]+)+\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)*\b|"
        r"\b[A-Z]\.-[A-Z]\.\s+(?:(?:van\s+(?:der\s+)?)?[A-Z][a-z]+)\b|"
        r"\b[A-Z]\.\s?[A-Z]\.\s[A-Z][a-z]+\b|"
        r"\b[A-Z]\.\s[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b"
    ),
    score=0.95,
)

InitialNameRecognizer = PatternRecognizer(
    supported_entity="PERSON",
    patterns=[INITIAL_NAME_PATTERN],
    name="InitialNameRecognizer",
)

CORP_EMAIL_PATTERN = Pattern(
    name="corp_email_pattern",
    regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    score=1.0,
)

CorpEmailRecognizer = PatternRecognizer(
    supported_entity="EMAIL_ADDRESS",
    patterns=[CORP_EMAIL_PATTERN],
    name="CorpEmailRecognizer",
)

ROBUST_PHONE_PATTERN = Pattern(
    name="robust_phone_pattern",
    regex=(
        r"(?<!\d)(?<!\d\.)"
        r"(?!\d{4}[\s.-]\d{4}[\s.-]\d{4}[\s.-]\d{4})"
        r"(?!\d{3}-\d{2}-\d{4})"
        r"(?!(?:\d{1,3}\.){3}\d{1,3})"
        r"(?:"
        r"(?:\+\d{1,3}[\s.-]*)?"
        r"(?:\(\d{1,4}\)|\d{1,4})[\s.-]*"
        r"(?:\(?\d{1,4}\)?[\s.-]*){1,3}"
        r"\d{3,5}"
        r"(?:[\s.-]*(?:ext|x|ext\.)[\s.-]*\d{1,5})?"
        r")"
        r"(?!\.\d)(?!\d)"
    ),
    score=0.85,
)

RobustPhoneRecognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[ROBUST_PHONE_PATTERN],
    name="RobustPhoneRecognizer",
)

STRICT_SSN_PATTERN = Pattern(
    name="strict_ssn_pattern",
    regex=r"\b\d{3}-\d{2}-\d{4}\b",
    score=1.0,
)

StrictSsnRecognizer = PatternRecognizer(
    supported_entity="US_SSN",
    patterns=[STRICT_SSN_PATTERN],
    name="StrictSsnRecognizer",
)

CORP_CREDIT_CARD_PATTERN = Pattern(
    name="corp_credit_card_pattern",
    regex=r"\b(?:\d{4}[-\s]){3}\d{4}\b|\b\d{16}\b",
    score=1.0,
)

CorpCreditCardRecognizer = PatternRecognizer(
    supported_entity="CREDIT_CARD",
    patterns=[CORP_CREDIT_CARD_PATTERN],
    name="CorpCreditCardRecognizer",
)

TARGET_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "ORGANIZATION",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
]

IGNORED_WORDS = {
    "RACI", "SWOT", "KPI", "SLA", "OKR", "OKRS", "PDF", "PPTX",
    "EMAIL", "SUBDOMAIN", "SSN", "ROSTER", "APPENDIX", "FALLBACK",
    "PRIMARY GATEWAY", "INFRASTRUCTURE NODES", "CREDIT CARD REF"
}

HIGH_RISK_VERIFICATION_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
]