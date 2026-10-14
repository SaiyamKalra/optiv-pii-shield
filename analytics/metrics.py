# analytics/metrics.py

import pandas as pd

from database.db import engine


def fetch_executive_kpis() -> dict:
    df_summary = pd.read_sql("SELECT * FROM document_summary", con=engine)
    if df_summary.empty:
        return {
            "total_documents": 0,
            "pass_rate_pct": 100.0,
            "total_pii_masked": 0,
        }

    total_docs = len(df_summary)
    passed_docs = int(df_summary["verification_passed"].sum())
    total_pii = int(df_summary["total_entities_detected"].sum())

    return {
        "total_documents": total_docs,
        "pass_rate_pct": round((passed_docs / total_docs) * 100, 2),
        "total_pii_masked": total_pii,
    }


def fetch_entity_distribution() -> pd.DataFrame:
    query = "SELECT entity_type, token FROM pii_detection_logs"
    df_logs = pd.read_sql(query, con=engine)
    if df_logs.empty:
        return pd.DataFrame()

    summary = (
        df_logs.groupby("entity_type").agg(occurrences=("token", "count")).reset_index()
    )
    return summary.sort_values(by="occurrences", ascending=False)


def fetch_confidence_score() -> pd.DataFrame:
    query = "SELECT entity_type, confidence_score FROM pii_detection_logs"
    df_logs = pd.read_sql(query, con=engine)

    if df_logs.empty:
        return pd.DataFrame()

    summary = (
        df_logs.groupby("entity_type")
        .agg(avg_confidence=("confidence_score", "mean"))
        .reset_index()
    )
    return summary.round(4)


def fetch_risk_exposure_matrix() -> pd.DataFrame:
    query = "SELECT file_id, entity_type, token FROM pii_detection_logs"
    df_logs = pd.read_sql(query, con=engine)
    if df_logs.empty:
        return pd.DataFrame()
    pivot = (
        df_logs.pivot_table(
            index="file_id",
            columns="entity_type",
            values="token",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
    )
    return pivot


def fetch_recent_pass_rates() -> pd.DataFrame:
    query = (
        "SELECT processed_at, verification_passed "
        "FROM document_summary "
        "WHERE verification_passed IS NOT NULL "
        "ORDER BY processed_at DESC LIMIT 10"
    )
    df = pd.read_sql(query, con=engine)

    if df.empty:
        return pd.DataFrame()

    df = df.sort_values("processed_at").reset_index(drop=True)
    df["pass_rate_pct"] = df["verification_passed"].astype(float) * 100
    df["document_number"] = range(1, len(df) + 1)

    return df[["document_number", "processed_at", "pass_rate_pct"]]


def get_conversion_matrix(raw_text: str, pii_outputs: list) -> dict:
    """Calculates data conversion and transformation metrics safely without depending on missing TokenOptimizer classmethods."""
    raw_text = raw_text or ""
    total_chars = len(raw_text)
    total_words = len(raw_text.split())
    estimated_raw_tokens = max(1, int(total_words * 1.3)) if total_words > 0 else 0

    total_redacted_entities = 0
    entity_type_counts = {}
    redacted_chars_count = 0

    for output in pii_outputs or []:
        entities = getattr(output, "detected_entities", []) or []
        for detail in entities:
            total_redacted_entities += 1

            # Safely handle both object and dict structures
            if hasattr(detail, "entity_type"):
                entity_type = detail.entity_type
                orig_val = getattr(detail, "original_value", "") or ""
            elif isinstance(detail, dict):
                entity_type = detail.get("entity_type", "UNKNOWN")
                orig_val = detail.get("original_value", "") or ""
            else:
                entity_type = "UNKNOWN"
                orig_val = ""

            entity_type_counts[entity_type] = entity_type_counts.get(entity_type, 0) + 1
            redacted_chars_count += len(orig_val)

    redaction_density_pct = (
        (redacted_chars_count / total_chars * 100) if total_chars > 0 else 0.0
    )

    return {
        "total_words_processed": total_words,
        "total_characters_processed": total_chars,
        "total_pii_entities_found": total_redacted_entities,
        "total_redacted_characters": redacted_chars_count,
        "redaction_density_percentage": round(redaction_density_pct, 2),
        "entity_breakdown": entity_type_counts,
        "estimated_raw_tokens": estimated_raw_tokens,
    }


def reading_coverage(chunks: list) -> dict:
    """How much of a document was read reliably, from what extraction reported.

    Items are words plus unread regions (ink OCR could not read), so a barcode counts
    against coverage even though it holds no words. A word is reliable unless OCR
    flagged it as low-confidence. A page with extraction errors was not inspected, so
    none of it is reliable.
    """
    total = reliable = 0
    for chunk in chunks or []:
        words = len((getattr(chunk, "raw_text", "") or "").split())
        unread = len(getattr(chunk, "unread_regions", None) or [])
        if getattr(chunk, "extraction_errors", None):
            total += max(words, 1) + unread
            continue
        doubtful = min(int(getattr(chunk, "low_confidence_word_count", 0) or 0), words)
        total += words + unread
        reliable += words - doubtful
    return {
        "total": total,
        "reliable": reliable,
        "coverage_pct": round(reliable / total * 100, 2) if total else 100.0,
    }