from presidio_analyzer import Pattern, PatternRecognizer

INITIAL_NAME_PATTERN=Pattern(
    name="initial_name_pattern",
    regex=r"\b[A-Z]\.\s[A-Z][a-z]+(?:-[A-Z][a-z]+)?\b",
    score=0.85,
)

InitialNameRecognizer=PatternRecognizer(
    supported_entity="PERSON",
    patterns=[INITIAL_NAME_PATTERN],
    name="InitialNameRecognizer",
)

CORP_EMAIL_PATTERN=Pattern(
    name="corp_email_pattern",
    regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    score=1.0,
)

CorpEmailRecognizer=PatternRecognizer(
    supported_entity="EMAIL_ADDRESS",
    patterns=[CORP_EMAIL_PATTERN],
    name="CorpEmailRecognizer",
)

ROBUST_PHONE_PATTERN = Pattern(
    name="robust_phone_pattern",
    regex=(
        r"(?:^|(?<=\s|\())"                     
        r"(?:\+\d{1,3}[\s.-]*)?"                 
        r"(?:\(\d{3,4}\)|\d{3,4})[\s.-]*"       
        r"\d{3,4}(?:[\s.-]*\d{3,4})?"           
        r"(?:[\s.-]*(?:ext|x|ext.)[\s.-]*\d{1,5})?" 
        r"(?=\b|\s|\)|,|$)"
    ),
    score=0.85,
)

RobustPhoneRecognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[ROBUST_PHONE_PATTERN],
    name="RobustPhoneRecognizer",
)

TARGET_ENTITIES=[
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "ORGANIZATION",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
]

IGNORED_ORGANIZATIONS = {"RACI", "SWOT", "KPI", "SLA", "OKR", "OKRS", "PDF", "PPTX"}

HIGH_RISK_VERIFICATION_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
]