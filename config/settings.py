from presidio_analyzer import Pattern, PatternRecognizer

# Person names with initials and hyphenated names
INITIAL_NAME_PATTERN = Pattern(
    name="initial_name_pattern",
    regex=(
        r"\b[A-Z][a-z]{1,30}(?:-[A-Z][a-z]{1,30})*"
        r"\s+[A-Z][a-z]{1,30}(?:-[A-Z][a-z]{1,30})*\b"
        r"|"
        r"\b[A-Z][a-z]{1,30}(?:-[A-Z][a-z]{1,30})*"
        r"\s+[A-Z]\.?\s+[A-Z][a-z]{1,30}\b"
        r"|"
        r"\b[A-Z]\.?\s*(?:[A-Z]\.?\s*)?"
        r"[A-Z][a-z]{1,30}\b"
    ),
    score=0.85,
)

InitialNameRecognizer = PatternRecognizer(
    supported_entity="PERSON",
    patterns=[INITIAL_NAME_PATTERN],
    name="InitialNameRecognizer",
)

# Flexible name pattern for complex full names in tabular formats
# (Handles middle initials, hyphenated surnames, and board role suffixes)
FLEXIBLE_NAME_PATTERN = Pattern(
    name="flexible_name_pattern",
    regex=(
        r"\b[A-Z][a-z]{1,30}"
        r"(?:\s+[A-Z]\.?)?"
        r"\s+[A-Z][a-z]{1,30}"
        r"(?:-[A-Z][a-z]{1,30})?"
        r"(?:\s*\((?:NED|CRO|CISO|CCO|CFO|Chair)\))?\b"
    ),
    score=0.85,
)

FlexibleNameRecognizer = PatternRecognizer(
    supported_entity="PERSON",
    patterns=[FLEXIBLE_NAME_PATTERN],
    name="FlexibleNameRecognizer",
    context=["chair", "member", "appointed", "contact", "executive", "director", 
        "head", "relationship owner", "deputy", "owner", "regulator", "lead"],
)

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

# Phone numbers with optional country code, area code, and extension
ROBUST_PHONE_PATTERN = Pattern(
    name="robust_phone_pattern",
    regex=(
        r"(?![A-Za-z]{2,10}-\d)"
        r"(?<![A-Za-z0-9])"
        r"(?!\d{4}[\s.-]\d{4}[\s.-]\d{4}[\s.-]\d{4}(?!\d))"
        r"(?:"
        r"\+\d{1,3}[\s.-]?"
        r"|(?:\(\+\d{1,3}\)\s?)"
        r")?"
        r"(?:\(\d{2,4}\)|\d{2,4})"
        r"[\s.-]?"
        r"\d{3,4}"
        r"[\s.-]?"
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

# Entities used during detection
TARGET_ENTITIES = [
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

IGNORED_WORDS = {
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
    "ISO", "COSO", "BCBS", "NIST", "DORA", "GDPR", "GLBA", "DPDP", 
    "UK DPA", "IIA", "PRA", "RCSA", "BASEL", "ORX", "NIST CSF",
    "ENTERPRISE RISK MANAGEMENT", "RISK DATA AGGREGATION",
    "OPERATIONAL RESILIENCE", "DIGITAL OPERATIONAL RESILIENCE ACT",
}

HIGH_RISK_VERIFICATION_ENTITIES = [
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

# Optimal NER threshold for ML models (GLiNER & spaCy)
MIN_NER_SCORE = 0.60  

# Strict pattern threshold for deterministic regex recognizers
MIN_PATTERN_SCORE = 0.85  

# Added GlinerRecognizer so _min_score_for() routes GLiNER spans correctly
NER_RECOGNIZER_PREFIXES = ("SpacyRecognizer", "GlinerRecognizer", "gliner")