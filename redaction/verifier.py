from models.schemas import PIIOutput


class RedactionVerifier:

    def verify(
        self,
        rescanned_results: list[PIIOutput],
        expected_pages: int,
    ) -> bool:
        """
        Pass only if every expected page was rescanned
        and no PII was detected.
        """

        if expected_pages <= 0:
            return False

        if len(rescanned_results) != expected_pages:
            return False

        page_numbers = set()

        for result in rescanned_results:

            if result.page_number in page_numbers:
                return False

            page_numbers.add(result.page_number)

            if result.status != "SUCCESS":
                return False
            if not result.verification_passed:
                return False
            if result.detected_entities:
                return False

        expected_page_numbers = set(range(1, expected_pages + 1))

        return page_numbers == expected_page_numbers
