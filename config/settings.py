"""Recognizers and Settings Configuration for PII Detection Pipeline."""


from presidio_analyzer import Pattern, PatternRecognizer

# =============================================================================
# RECOGNIZER PATTERNS & INSTANCES
# =============================================================================

# Corporate email addresses
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

# Phone numbers with optional country code, area code, extension, and multiline table support
ROBUST_PHONE_PATTERN = Pattern(
    name="robust_phone_pattern",
    regex=(
        r"(?![A-Za-z]{2,10}-\d)"
        r"(?<![A-Za-z0-9])"
        r"(?!\d{4}[\s.-]\d{4}[\s.-]\d{4}[\s.-]\d{4}(?!\d))"
        r"(?:"
        r"\+\d{1,3}[\s\n.-]?"
        r"|(?:\(\+\d{1,3}\)[\s\n]?)"
        r")?"
        r"(?:\(\d{2,4}\)|\d{2,4})"
        r"[\s\n.-]+"  # Match spaces OR newlines between lines in tables
        r"\d{3,4}"
        r"[\s\n.-]?"
        r"\d{3,5}"
        r"(?:\s*(?:ext\.?|x)\s*\d{1,5})?"
        r"(?![A-Za-z0-9])"
    ),
    score=0.85,
)

RobustPhoneRecognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[ROBUST_PHONE_PATTERN],
    name="RobustPhoneRecognizer",
)

# US Social Security numbers
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

# Credit card numbers
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


class ContextualNameRecognizer(PatternRecognizer):
    """Matches full names, all-caps names, and initialisms when near role context words."""

    def __init__(self) -> None:
        name_patterns = [
            # Two or three capitalized/uppercase words (e.g., "Desmond N. Quarshie", "Mireille Dubois-Kane")
            Pattern(
                name="full_name_context",
                regex=r"\b[A-Z][a-zA-Z'-]+\s+(?:[A-Z]\.\s+)?[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)?\b",
                score=0.75,
            ),
            # Single surname with initial (e.g., "D. Quarshie", "M. Dubois")
            Pattern(
                name="initial_surname_context",
                regex=r"\b[A-Z]\.\s+[A-Z][a-zA-Z'-]+\b",
                score=0.70,
            ),
        ]

        context_terms = [
            "relationship owner",
            "owner",
            "deputy",
            "contact",
            "lead",
            "head",
            "director",
            "manager",
            "approved by",
            "sign-off",
            "chair",  # Added for table headers like "CHAIR"
            "chair contact",  # Added for table headers like "CHAIR CONTACT"
        ]

        super().__init__(
            supported_entity="PERSON",
            patterns=name_patterns,
            name="ContextualNameRecognizer",
            context=context_terms,
        )


# Organization-specific identifiers
EmployeeIdRecognizer = PatternRecognizer(
    supported_entity="EMPLOYEE_ID",
    name="EmployeeIdRecognizer",
    patterns=[
        Pattern(
            name="employee_id",
            regex=r"\bEMP-\d{4,8}\b",
            score=0.99,
        )
    ],
    context=["employee", "employee id", "staff id"],
)

DirectorIdRecognizer = PatternRecognizer(
    supported_entity="DIRECTOR_ID",
    name="DirectorIdRecognizer",
    patterns=[
        Pattern(
            name="director_id",
            regex=r"\bDIR-\d{4,8}\b",
            score=0.99,
        )
    ],
    context=["director", "director id"],
)

ExceptionIdRecognizer = PatternRecognizer(
    supported_entity="EXCEPTION_ID",
    name="ExceptionIdRecognizer",
    patterns=[
        Pattern(
            name="exception_id",
            regex=r"\bEXC-\d{3,8}\b",
            score=0.99,
        )
    ],
    context=["exception", "exception id", "exception reference"],
)

DocumentReferenceRecognizer = PatternRecognizer(
    supported_entity="DOCUMENT_REFERENCE",
    name="DocumentReferenceRecognizer",
    patterns=[
        Pattern(
            name="policy_reference",
            regex=r"\bGRP-POL-\d{2,8}\b",
            score=0.99,
        )
    ],
    context=["policy reference", "document reference", "policy id"],
)

# =============================================================================
# PIPELINE CONFIGURATION & TARGET ENTITIES
# =============================================================================

TARGET_ENTITIES: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "ORGANIZATION",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "EMPLOYEE_ID",
    "DIRECTOR_ID",
    "EXCEPTION_ID",
    "DOCUMENT_REFERENCE",
]

IGNORED_WORDS: set[str] = {
    "RACI",
    "SWOT",
    "KPI",
    "SLA",
    "OKR",
    "OKRS",
    "PDF",
    "PPTX",
    "EMAIL",
    "SUBDOMAIN",
    "SSN",
    "ROSTER",
    "APPENDIX",
    "FALLBACK",
    "PRIMARY GATEWAY",
    "INFRASTRUCTURE NODES",
    "CREDIT CARD REF",
    "RACI MATRIX",
    "COMPLIANCE SIGN-OFF",
    "ROSTER APPENDIX",
    "SLA & KPI",
    "ISO",
    "COSO",
    "BCBS",
    "NIST",
    "DORA",
    "GDPR",
    "GLBA",
    "DPDP",
    "UK DPA",
    "IIA",
    "PRA",
    "RCSA",
    "BASEL",
    "ORX",
    "NIST CSF",
    "ENTERPRISE RISK MANAGEMENT",
    "RISK DATA AGGREGATION",
    "OPERATIONAL RESILIENCE",
    "DIGITAL OPERATIONAL RESILIENCE ACT",
}

HIGH_RISK_VERIFICATION_ENTITIES: list[str] = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "EMPLOYEE_ID",
    "DIRECTOR_ID",
    "EXCEPTION_ID",
    "DOCUMENT_REFERENCE",
    "IP_ADDRESS",
]

# Score Thresholds
MIN_NER_SCORE: float = 0.60
MIN_PATTERN_SCORE: float = 0.85
NER_RECOGNIZER_PREFIXES: tuple[str, ...] = (
    "SpacyRecognizer",
    "GlinerRecognizer",
    "gliner",
)