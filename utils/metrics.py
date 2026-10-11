# utils/metrics.py


from models.schemas import PIIOutput


def calculate_pii_precision(findings: list, ground_truth_pii: set[str]) -> dict:
    """
    Calculates False Positives, True Positives, and Precision %.
    ground_truth_pii: Set of known actual PII strings in the test document.
    """
    total_detected = len(findings)
    if total_detected == 0:
        return {"precision_pct": 100.0, "false_positives": 0, "fp_rate_pct": 0.0}

    true_positives = 0
    false_positives = 0
    fp_items = []

    for item in findings:
        val = item.original_value.strip()
        if val in ground_truth_pii:
            true_positives += 1
        else:
            false_positives += 1
            fp_items.append((val, item.entity_type, item.page_number))

    precision = (true_positives / total_detected) * 100
    fp_rate = (false_positives / total_detected) * 100

    return {
        "total_detected": total_detected,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "precision_pct": round(precision, 2),
        "fp_rate_pct": round(fp_rate, 2),
        "false_positive_items": fp_items,
    }


def calculate_word_coverage(raw_text: str, pii_outputs: list[PIIOutput]) -> dict:
    """Calculates word preservation and redaction density across processed chunks."""
    total_words = len(raw_text.split())
    if total_words == 0:
        return {"total_words": 0, "redacted_words": 0, "preservation_pct": 0.0}

    redacted_words_count = 0
    for page in pii_outputs:
        for entity in page.detected_entities:
            words_in_span = len(entity.original_value.strip().split())
            redacted_words_count += words_in_span

    clean_words_preserved = total_words - redacted_words_count
    preservation_pct = (clean_words_preserved / total_words) * 100
    redact_density_pct = (redacted_words_count / total_words) * 100

    return {
        "total_words": total_words,
        "redacted_words": redacted_words_count,
        "preserved_clean_words": clean_words_preserved,
        "document_preservation_pct": round(preservation_pct, 2),
        "redact_density_pct": round(redact_density_pct, 2),
    }