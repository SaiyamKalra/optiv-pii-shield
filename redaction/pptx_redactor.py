from redaction.office_xml_utils import redact_office_archive


class PPTXRedactor:
    def redact(self, input_pptx, output_pptx, chunks, pii_results):
        return redact_office_archive(
            input_pptx,
            output_pptx,
            pii_results,
        )
