from redaction.office_redactor import OfficeRedactor


class PPTXRedactor(OfficeRedactor):
    """Masks PII text in every part of a .pptx and blacks it out inside its pictures."""
