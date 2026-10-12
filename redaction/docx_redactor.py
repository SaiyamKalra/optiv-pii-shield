from redaction.office_redactor import OfficeRedactor


class DOCXRedactor(OfficeRedactor):
    """Masks PII text in every part of a .docx and blacks it out inside its images."""
