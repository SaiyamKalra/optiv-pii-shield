from redaction.office_xml_utils import redact_office_archive


class DOCXRedactor:
    def redact(self, input_docx, output_docx, chunks, pii_results):
        return redact_office_archive(
            input_docx,
            output_docx,
            pii_results,
        )
