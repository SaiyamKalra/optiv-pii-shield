"""Recognizers and Settings Configuration for PII Detection Pipeline."""


import re

from presidio_analyzer import Pattern, PatternRecognizer


class FilteredPatternRecognizer(PatternRecognizer):
    """PatternRecognizer whose matches must also pass keep(text, start, end).

    Presidio's own `context` only raises a score; these checks drop a match outright.
    """

    def keep(self, text: str, start: int, end: int) -> bool:
        return True

    def analyze(self, text, entities, nlp_artifacts=None, regex_flags=None):
        results = super().analyze(text, entities, nlp_artifacts, regex_flags)
        return [r for r in results if self.keep(text, r.start, r.end)]


class ContextPatternRecognizer(FilteredPatternRecognizer):
    """Keeps a match only if its label comes directly before it, with only spaces or
    punctuation in between ("passport TK4820913", "DOB 14 July 1991", "date of birth,
    3 August 1996"). A label further back does not count: "PESEL 890412xxxxx +48 512555"
    must not turn the phone's digits into an ID."""

    def __init__(self, *args, required_context: list[str], window: int = 40, **kwargs):
        super().__init__(*args, **kwargs)
        self.required = re.compile(
            r"(?<!\w)(?:" + "|".join(map(re.escape, required_context)) + r")[\s:,#.()\-]*$",
            re.IGNORECASE,
        )
        self.window = window

    def keep(self, text, start, end):
        return bool(self.required.search(text[max(0, start - self.window) : start]))


class DigitCountPatternRecognizer(FilteredPatternRecognizer):
    """Keeps a match only if it holds between min_digits and max_digits digits."""

    def __init__(self, *args, min_digits: int, max_digits: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_digits, self.max_digits = min_digits, max_digits

    def keep(self, text, start, end):
        return self.min_digits <= sum(ch.isdigit() for ch in text[start:end]) <= self.max_digits


# =============================================================================
# RECOGNIZER PATTERNS & INSTANCES
# =============================================================================

# Corporate email addresses
CORP_EMAIL_PATTERN = Pattern(
    name="corp_email_pattern",
    # One optional space around "@": OCR splits "n.osadebe @cadencefg.com".
    regex=r"\b[A-Za-z0-9._%+-]+ ?@ ?[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
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

# International numbers: "+33 1 5555 0144", "+91 98860 55107", "+1 (555) 0114".
INTERNATIONAL_PHONE_PATTERN = Pattern(
    name="international_phone_pattern",
    regex=r"(?<![\w+])\+\d{1,3}(?:[ .-]?\(?\d{1,5}\)?){2,5}(?![\w-])",
    score=0.9,
)

RobustPhoneRecognizer = DigitCountPatternRecognizer(
    supported_entity="PHONE_NUMBER",
    patterns=[ROBUST_PHONE_PATTERN, INTERNATIONAL_PHONE_PATTERN],
    name="RobustPhoneRecognizer",
    min_digits=7,
    max_digits=15,
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

# Personal identifiers that need a label right before them.
PassportRecognizer = ContextPatternRecognizer(
    supported_entity="PASSPORT",
    name="PassportRecognizer",
    patterns=[Pattern(name="passport", regex=r"\b[A-Z]{1,2}\d{6,9}\b", score=0.95)],
    required_context=["passport", "passport no", "passport number"],
)

NationalIdRecognizer = ContextPatternRecognizer(
    supported_entity="NATIONAL_ID",
    name="NationalIdRecognizer",
    # Masked forms count too ("PESEL 890412xxxxx").
    patterns=[Pattern(name="national_id", regex=r"\b\d{6,12}[xX*]{0,6}(?![\w*])", score=0.95)],
    required_context=["PESEL", "Aadhaar", "Aadhar", "national id", "national insurance", "NIN"],
)

_MONTH = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
DateOfBirthRecognizer = ContextPatternRecognizer(
    supported_entity="DATE_OF_BIRTH",
    name="DateOfBirthRecognizer",
    patterns=[
        Pattern(name="dob_day_month", regex=rf"\b\d{{1,2}}\s+{_MONTH}\.?\s+\d{{4}}\b", score=0.95),
        Pattern(name="dob_month_day", regex=rf"\b{_MONTH}\.?\s+\d{{1,2}},?\s+\d{{4}}\b", score=0.95),
        Pattern(name="dob_numeric", regex=r"\b\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})\b", score=0.95),
    ],
    required_context=["DOB", "D.O.B", "date of birth", "birth date", "birthdate", "born"],
)

# Indian PAN: 5 letters, 4 digits, 1 letter (AQBPR7741L). Distinctive on its own.
PanRecognizer = PatternRecognizer(
    supported_entity="PAN",
    name="PanRecognizer",
    patterns=[Pattern(name="pan", regex=r"\b[A-Z]{5}\d{4}[A-Z]\b", score=0.9)],
    context=["pan", "permanent account number"],
)

# Contractor/vendor staff IDs (MER-IN-4471).
VendorIdRecognizer = PatternRecognizer(
    supported_entity="VENDOR_ID",
    name="VendorIdRecognizer",
    patterns=[Pattern(name="vendor_id", regex=r"\b[A-Z]{3}-[A-Z]{2}-\d{4}\b", score=0.95)],
    context=["vendor", "contractor", "staff id"],
)

_STREET = (
    r"(?:Street|St|Lane|Ln|Road|Rd|Avenue|Ave|Court|Ct|Crescent|Cres|Drive|Dr|Boulevard|Blvd|"
    r"Way|Place|Pl|Terrace|Close|Square|Sq|Parkway|Pkwy|Highway|Hwy)"
)
StreetAddressRecognizer = PatternRecognizer(
    supported_entity="STREET_ADDRESS",
    name="StreetAddressRecognizer",
    patterns=[
        # "88 Willowbank Court", "2214 Palmetto Crescent"
        Pattern(
            name="house_number_street",
            regex=rf"\b\d{{1,5}}\s+(?:[A-Z][a-z]+\s+){{1,3}}{_STREET}\b\.?",
            score=0.9,
        ),
        # "Hoboken, NJ 07030", "Coral Gables, FL 33134"
        Pattern(
            name="city_state_zip",
            regex=r"\b(?:[A-Z][a-z]+\s){0,2}[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b",
            score=0.9,
        ),
    ],
)

# Not registered: policy numbers and exception IDs are not personal data. Kept so they
# can be switched back on in pii/pipeline.py.
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

# ORGANIZATION and bare LOCATION (cities, countries) are not personal data and were
# ~40% of false positives; home addresses are covered by STREET_ADDRESS instead.
TARGET_ENTITIES: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "STREET_ADDRESS",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "EMPLOYEE_ID",
    "DIRECTOR_ID",
    "VENDOR_ID",
    "PASSPORT",
    "PAN",
    "NATIONAL_ID",
    "DATE_OF_BIRTH",
]

HIGH_RISK_VERIFICATION_ENTITIES: list[str] = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "EMPLOYEE_ID",
    "DIRECTOR_ID",
    "VENDOR_ID",
    "PASSPORT",
    "PAN",
    "NATIONAL_ID",
    "DATE_OF_BIRTH",
    "STREET_ADDRESS",
    "IP_ADDRESS",
]

# Score Thresholds
MIN_NER_SCORE: float = 0.60
MIN_PATTERN_SCORE: float = 0.85
NER_RECOGNIZER_PREFIXES: tuple[str, ...] = ("SpacyRecognizer",)
