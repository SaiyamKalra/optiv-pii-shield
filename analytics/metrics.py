import pandas as pd
from database.db import engine

def fetch_executive_kpis() -> dict:
    df_summary = pd.read_sql("SELECT * FROM document_summary", con=engine)
    if df_summary.empty:
        return {
            "total_documents": 0,
            "pass_rate_pct": 100,
            "total_pii_masked": 0,
        }

    total_docs = len(df_summary)
    passed_docs = int(df_summary['verification_passed'].sum())
    total_pii = int(df_summary['total_entities_detected'].sum())
     
    return {
        "total_documents": total_docs,
        "pass_rate_pct": round((passed_docs / total_docs) * 100, 2),
        "total_pii_masked": total_pii,
    }

def fetch_entity_distribution() -> pd.DataFrame:
    query = "SELECT entity_type,token FROM pii_detection_logs"
    df_logs = pd.read_sql(query, con=engine)
    if df_logs.empty:
        return pd.DataFrame()
    
    summary = (
        df_logs.groupby("entity_type").agg(occurrences=("token", "count")).reset_index()
    )
    return summary.sort_values(by="occurrences", ascending=False)

def fetch_confidence_score() -> pd.DataFrame:
    query = "SELECT entity_type,confidence_score FROM pii_detection_logs"
    df_logs = pd.read_sql(query, con=engine)
    
    if df_logs.empty:
        return pd.DataFrame()
    
    summary = (
        df_logs.groupby("entity_type").agg(avg_confidence=("confidence_score", "mean")).reset_index()
    )
    return summary.round(4)

def fetch_risk_exposure_matrix() -> pd.DataFrame:
    query = "SELECT file_id,entity_type,token FROM pii_detection_logs"
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
    query = "SELECT processed_at,verification_passed FROM document_summary WHERE verification_passed IS NOT NULL ORDER BY processed_at DESC LIMIT 10"
    df = pd.read_sql(query, con=engine)

    if df.empty:
        return pd.DataFrame()
    
    df = df.sort_values('processed_at').reset_index(drop=True)
    df["pass_rate_pct"] = (
        df["verification_passed"].astype(float) * 100
    )

    df["document_number"] = range(1, len(df) + 1)

    return df[["document_number", "processed_at", "pass_rate_pct"]]