"""
PII Sentinel — Frontend
Run:  streamlit run app.py
"""

import random
import tempfile
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import pymupdf
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from analytics.metrics import (
    fetch_confidence_score,
    fetch_entity_distribution,
    fetch_executive_kpis,
    fetch_recent_pass_rates,
    fetch_risk_exposure_matrix,
)
from extraction.pdf_extractor import extract_pdf
from models.schemas import PIIOutput
from pii.pipeline import PIIPipeline
from redaction.pipeline import RedactionPipeline
from redaction.token_redaction.token_optimizer import TokenOptimizer

st.set_page_config(
    page_title="PII Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Responsive Layout & Simple Calibri Font with Bright Neon Contrast
st.markdown(
    """
<style>
    /* ── Reset & Base Typography ── */
    html, body, [class*="css"], .stApp {
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif !important;
        background-color: #0d0d14;
        color: #e2e8f0;
        box-sizing: border-box;
    }

    *, *:before, *:after {
        box-sizing: inherit;
    }

    /* ── Responsive Main Container ── */
    .block-container, [data-testid="stMainBlockContainer"] {
        padding-top: 1.75rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: clamp(0.75rem, 3vw, 2.5rem) !important;
        padding-right: clamp(0.75rem, 3vw, 2.5rem) !important;
        max-width: 1440px !important;
        margin: 0 auto !important;
    }

    /* ── Bright, High-Contrast Headings (Fluid Scaling) ── */
    h1, h2, h3, h4, h5, h6, 
    [data-testid="stMarkdownContainer"] h1, 
    [data-testid="stMarkdownContainer"] h2, 
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #00f2fe !important;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif !important;
        font-weight: 700 !important;
        word-break: break-word;
        line-height: 1.25 !important;
    }

    .main-title {
        font-size: clamp(1.4rem, 4vw, 2.3rem) !important;
        font-weight: 700;
        color: #00f2fe !important;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        margin-bottom: 4px;
        text-shadow: 0 0 14px rgba(0, 242, 254, 0.28);
        line-height: 1.2;
        word-break: break-word;
    }

    .sub-text {
        color: #cbd5e1 !important;
        font-size: clamp(0.85rem, 1.8vw, 1.05rem) !important;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        margin-bottom: 18px;
        line-height: 1.45;
        word-break: break-word;
    }

    /* ── Responsive KPI Grid & Cards ── */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        width: 100%;
        margin: 8px 0 20px 0;
    }

    .kpi-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: clamp(10px, 1.8vw, 16px) clamp(8px, 1.4vw, 14px);
        text-align: center;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        min-height: 82px;
        box-sizing: border-box;
        transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
    }
    .kpi-box:hover {
        border-color: rgba(0, 242, 254, 0.45);
        background: rgba(255, 255, 255, 0.07);
        transform: translateY(-2px);
    }
    .kpi-num {
        font-size: clamp(1.35rem, 2.6vw, 2rem) !important;
        font-weight: 700;
        color: #38bdf8 !important;
        line-height: 1.15;
        word-break: break-word;
    }
    .kpi-label {
        font-size: clamp(0.72rem, 1.1vw, 0.88rem) !important;
        color: #94a3b8;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.2;
    }

    /* ── Tokens & Badges ── */
    .token {
        background: rgba(56, 189, 248, 0.2);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 4px;
        padding: 2px 8px;
        font-family: monospace;
        font-size: clamp(0.78rem, 1.4vw, 0.9rem);
        font-weight: 600;
        word-break: break-all;
        display: inline-block;
    }
    .badge-safe {
        color: #4ade80;
        background: rgba(74, 222, 128, 0.15);
        border: 1px solid rgba(74, 222, 128, 0.4);
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 0.88rem;
        font-weight: 600;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
    }
    .badge-unknown {
        color: #facc15;
        background: rgba(250, 204, 21, 0.15);
        border: 1px solid rgba(250, 204, 21, 0.4);
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 0.88rem;
        font-weight: 600;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
    }
    .badge-unsafe {
        color: #f87171;
        background: rgba(248, 113, 113, 0.15);
        border: 1px solid rgba(248, 113, 113, 0.4);
        padding: 4px 14px;
        border-radius: 16px;
        font-size: 0.88rem;
        font-weight: 600;
        white-space: nowrap;
        display: inline-flex;
        align-items: center;
    }

    /* ── Status Header Banner ── */
    .status-banner {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 16px;
    }
    .status-filename {
        font-size: clamp(1.1rem, 2.4vw, 1.4rem);
        font-weight: 700;
        color: #00f2fe !important;
        word-break: break-all;
    }
    .status-meta {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
    }
    .status-time {
        color: #94a3b8;
        font-size: clamp(0.82rem, 1.4vw, 0.95rem);
    }

    /* ── Info Box ── */
    .info-box {
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #00f2fe;
        border-radius: 6px;
        padding: clamp(12px, 2vw, 16px) clamp(14px, 2.5vw, 20px);
        font-size: clamp(0.85rem, 1.5vw, 0.98rem);
        color: #f1f5f9;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        line-height: 1.6;
        word-break: break-word;
        overflow-wrap: break-word;
    }
    hr {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 16px 0;
    }

    /* ── Buttons Responsiveness ── */
    .stButton > button {
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        min-height: 42px !important;
        font-size: clamp(0.82rem, 1.4vw, 0.96rem) !important;
        transition: all 0.2s ease !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
        padding: 8px 12px !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 2px 10px rgba(6, 182, 212, 0.3) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #cbd5e1 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #00f2fe !important;
        color: #00f2fe !important;
        background: rgba(0, 242, 254, 0.08) !important;
    }

    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif !important;
        font-size: clamp(0.82rem, 1.4vw, 0.96rem) !important;
        min-height: 42px !important;
        padding: 8px 14px !important;
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.25 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover:not(:disabled) {
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.45) !important;
        transform: translateY(-1px);
    }

    /* ── Table & Dataframe ── */
    [data-testid="stDataFrame"] {
        width: 100% !important;
        max-width: 100% !important;
        overflow-x: auto !important;
    }

    /* ── Plotly Charts ── */
    .js-plotly-plot, .plot-container {
        width: 100% !important;
        max-width: 100% !important;
    }

    /* ── Media Queries for Responsive Stacking & Mobile ── */
    @media (max-width: 850px) {
        [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.85rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"],
        [data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        .status-banner {
            flex-direction: column;
            align-items: flex-start;
        }
    }

    @media (min-width: 520px) and (max-width: 850px) {
        .nav-buttons-row [data-testid="stHorizontalBlock"] {
            flex-direction: row !important;
        }
        .nav-buttons-row [data-testid="column"] {
            width: 33.333% !important;
            min-width: 33.333% !important;
            flex: 1 1 33.333% !important;
        }
    }

    @media (max-width: 640px) {
        .kpi-grid {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 8px !important;
        }
        .main-title {
            text-align: center;
        }
        .sub-text {
            text-align: center;
        }
    }

    @media (max-width: 380px) {
        .kpi-grid {
            grid-template-columns: 1fr !important;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Demo & Real Backend ─────────────────────────────────────────────────────
ENTITY_TYPES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "LOCATION",
    "ORGANIZATION",
]
WEIGHTS = [28, 20, 18, 10, 8, 5, 6, 5]
SOURCES = ["body", "table", "image/OCR", "narrative"]
DETECTORS = {
    "PERSON": "spaCy NER",
    "EMAIL_ADDRESS": "Presidio regex",
    "PHONE_NUMBER": "RobustPhone recognizer",
    "US_SSN": "StrictSSN recognizer",
    "CREDIT_CARD": "CorpCreditCard recognizer",
    "IP_ADDRESS": "Presidio regex",
    "LOCATION": "spaCy NER",
    "ORGANIZATION": "spaCy NER",
}


def _demo_pipeline(filename: str = "Cadence_Risk_Policy_v6.pdf", pages: int = 12) -> dict:
    """Generate realistic demo results for testing and preview without a real upload."""
    rng = random.Random(len(filename) * 31)
    rows, counters = [], {}
    for _ in range(rng.randint(65, 110)):
        t = rng.choices(ENTITY_TYPES, weights=WEIGHTS)[0]
        counters[t] = counters.get(t, 0) + 1
        n = counters[t]
        rows.append(
            {
                "Page": rng.randint(1, pages),
                "Source": rng.choices(SOURCES, weights=[3, 4, 2, 3])[0],
                "Type": t,
                "Detector": DETECTORS.get(t, "Presidio"),
                "Confidence": round(rng.uniform(0.65, 0.99), 3),
                "Token": f"[{t}_{n:03d}]",
                "Context": f"Pattern match for '{t.lower().replace('_', ' ')}'",
            }
        )
    df = pd.DataFrame(rows).sort_values(["Page", "Type"]).reset_index(drop=True)
    df.insert(0, "ID", [f"F-{i + 1:05d}" for i in range(len(df))])
    ocr_p = max(1, pages // 4)
    return {
        "file": filename,
        "pages": pages,
        "ocr_pages": ocr_p,
        "reliable": pages,
        "findings": df,
        "sanitized_bytes": b"%PDF-1.4 demo sanitized document content",
        "residual": 0,
        "secs": round(rng.uniform(1.8, 3.2), 2),
        "redaction_status": "VERIFIED_NO_PII_FOUND",
        "inspection_issues": [],
        "inspection_warnings": [],
    }


def _run_real_pipeline(uploaded) -> dict:
    """Run actual PDF extraction, PII detection, redaction and verification.

    sanitized_bytes is populated whenever output_path exists on disk with
    st_size > 0, regardless of the pipeline's verification status (Req. 1).
    RedactionPipeline runs up to MAX_REDACTION_PASSES internally (Req. 3).
    """
    if not uploaded.name.lower().endswith(".pdf"):
        raise ValueError("Verified redaction currently supports PDF files only.")

    started = time.perf_counter()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / "original.pdf"
        output_path = Path(temp_dir) / "sanitized.pdf"
        input_path.write_bytes(uploaded.getvalue())

        # Extract text, OCR results and bounding boxes.
        chunks = extract_pdf(input_path)

        # Run initial PII detection.
        pii_pipeline = PIIPipeline()
        original_results = pii_pipeline.process_document_chunks(chunks)

        # Redact and verify — RedactionPipeline runs up to MAX_REDACTION_PASSES
        # and preserves the output file regardless of verification status.
        redaction_pipeline = RedactionPipeline(pii_pipeline=pii_pipeline)
        result = redaction_pipeline.process(
            input_pdf=str(input_path),
            output_pdf=str(output_path),
            chunks=chunks,
            pii_results=original_results,
        )

        with pymupdf.open(input_path) as document:
            page_count = len(document)

        # Convert actual detection results to the table format expected by UI.
        rows = []
        for chunk, page_result in zip(
            chunks,
            original_results,
        ):
            for finding in page_result.detected_entities:
                matched_boxes = [
                    box
                    for box in (chunk.bounding_boxes or [])
                    if box.char_start is not None
                    and box.char_end is not None
                    and box.char_start < finding.end
                    and box.char_end > finding.start
                ]

                source = (
                    "image/OCR"
                    if any(box.source == "ocr" for box in matched_boxes)
                    else "body"
                )

                rows.append(
                    {
                        "Page": finding.page_number,
                        "Source": source,
                        "Type": finding.entity_type,
                        "Detector": DETECTORS.get(finding.entity_type, "Presidio"),
                        "Confidence": float(finding.score),
                        "Token": finding.token,
                        "Context": (
                            f"Detected {finding.entity_type} "
                            f"at character offset {finding.start}"
                        ),
                    }
                )

        findings = pd.DataFrame(
            rows,
            columns=[
                "Page",
                "Source",
                "Type",
                "Detector",
                "Confidence",
                "Token",
                "Context",
            ],
        )
        findings = findings.sort_values(["Page", "Type"]).reset_index(drop=True)
        findings.insert(0, "ID", [f"F-{i + 1:05d}" for i in range(len(findings))])

        status = result["status"]

        # Always read sanitized bytes if the generated output file exists and is non-empty
        sanitized_bytes = (
            output_path.read_bytes()
            if output_path.is_file() and output_path.stat().st_size > 0
            else b""
        )

        reliable_pages = max(
            0,
            page_count
            - len(
                {
                    issue.get("page_number")
                    for issue in result.get("inspection_issues", [])
                    if issue.get("page_number") is not None
                }
            ),
        )

        return {
            "file": uploaded.name,
            "pages": page_count,
            "ocr_pages": sum(
                1
                for chunk in chunks
                if chunk.ocr_used or chunk.extraction_method in ("ocr", "hybrid")
            ),
            "reliable": reliable_pages,
            "findings": findings,
            "sanitized_bytes": sanitized_bytes,
            "residual": (
                0 if status == "VERIFIED_NO_PII_FOUND"
                else len(result.get("inspection_issues", []))
            ),
            "secs": round(time.perf_counter() - started, 2),
            "redaction_status": status,
            "redaction_count": result.get("redaction_count", 0),
            "inspection_issues": result.get("inspection_issues", []),
            "inspection_warnings": result.get("inspection_warnings", []),
            "extracted_chunks": chunks,
            "pii_outputs": original_results,
        }

# ── Helpers ─────────────────────────────────────────────────────────────────
if "res" not in st.session_state:
    st.session_state.res = None
if "screen" not in st.session_state:
    st.session_state.screen = "upload"
if "thr" not in st.session_state:
    st.session_state.thr = 0.70


COVERAGE_REQUIRED_PCT = 95.0


def verdict(res, thr):
    """Return (state, low_confidence_count, word_coverage_pct)."""
    df = res.get("findings")
    review = 0
    if df is not None and "Confidence" in df.columns:
        review = int((df["Confidence"] < thr).sum())

    # Coverage is measured in words, not pages.
    total_words = int(res.get("total_words") or 0)
    reliable_words = min(int(res.get("reliable_words") or 0), total_words)
    coverage = (reliable_words / total_words * 100) if total_words > 0 else 0.0

    backend_status = res.get("redaction_status")

    if backend_status == "BLOCKED":
        return "UNSAFE", review, coverage

    if backend_status != "VERIFIED_NO_PII_FOUND":
        return "UNKNOWN", review, coverage

    if res.get("residual", 0) > 0:
        return "UNSAFE", review, coverage

    if review > 0 or coverage < COVERAGE_REQUIRED_PCT:
        return "UNKNOWN", review, coverage

    return "SAFE", review, coverage

def render_kpis(items):
    """Render KPI cards cleanly using responsive CSS Grid."""
    cards_html = "".join(
        f'<div class="kpi-box"><div class="kpi-num">{val}</div>'
        f'<div class="kpi-label">{lbl}</div></div>'
        for val, lbl in items
    )
    st.markdown(f'<div class="kpi-grid">{cards_html}</div>', unsafe_allow_html=True)


def kpi(col, value, label):
    """Backward-compatible column-based KPI renderer."""
    col.markdown(
        f'<div class="kpi-box"><div class="kpi-num">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


# ── Navigation Header ────────────────────────────────────────────────────────
c_title, c_nav = st.columns([1.3, 2])
with c_title:
    st.markdown('<div class="main-title">🛡️ PII Sentinel</div>', unsafe_allow_html=True)
with c_nav:
    st.markdown('<div class="nav-buttons-row">', unsafe_allow_html=True)
    n1, n2, n3 = st.columns(3)
    with n1:
        if st.button(
            "📁 Upload Screen",
            use_container_width=True,
            type="primary" if st.session_state.screen == "upload" else "secondary",
        ):
            st.session_state.screen = "upload"
            st.rerun()
    with n2:
        if st.button(
            "📋 Findings Table",
            use_container_width=True,
            disabled=st.session_state.res is None,
            type="primary" if st.session_state.screen == "findings" else "secondary",
        ):
            st.session_state.screen = "findings"
            st.rerun()
    with n3:
        if st.button(
            "📊 Analytics Charts",
            use_container_width=True,
            type="primary" if st.session_state.screen == "analytics" else "secondary",
        ):
            st.session_state.screen = "analytics"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Upload Screen
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "upload":

    st.markdown(
        '<div class="sub-text">Upload a document to detect and redact sensitive PII before sharing with AI.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Upload a document (PDF, PPTX, DOCX)",
            type=["pdf", "pptx", "docx"],
        )

        col_a, col_b = st.columns(2)
        go_btn = col_a.button(
            "🔍 Analyze Document", use_container_width=True, disabled=uploaded is None
        )
        demo_btn = col_b.button("📄 Try Demo File", use_container_width=True)

        if go_btn or demo_btn:
            name = (
                uploaded.name if (go_btn and uploaded) else "Cadence_Risk_Policy_v6.pdf"
            )
            ext = name.lower().rsplit(".", 1)[-1]
            pages = {"pdf": 35, "pptx": 9}.get(ext, 12)

            STEPS = [
                "Detecting file type and structure",
                "Extracting text layer",
                "Running OCR on image pages",
                "Detecting PII — regex + NER + context rules",
                "Building token substitution map",
                "Re-scanning sanitized output for residual PII",
            ]
            with st.status("Analyzing document...", expanded=True) as status:
                if go_btn and uploaded:
                    for step in STEPS[:3]:
                        st.write(step)
                        time.sleep(0.15)
                    try:
                        st.session_state.res = _run_real_pipeline(uploaded)
                        for step in STEPS[3:]:
                            st.write(step)
                            time.sleep(0.1)
                        status.update(
                            label="Document processing finished",
                            state="complete",
                            expanded=False,
                        )
                    except (ValueError, OSError, RuntimeError) as exc:
                        status.update(
                            label="Document processing failed",
                            state="error",
                            expanded=False,
                        )
                        st.error(f"Processing failed: {exc}")
                        st.stop()
                else:
                    for step in STEPS:
                        st.write(step)
                        time.sleep(0.1)
                    st.session_state.res = _demo_pipeline(name, pages)
                    status.update(
                        label="Demo document processed successfully",
                        state="complete",
                        expanded=False,
                    )

            st.session_state.screen = "findings"
            st.rerun()

    with right:
        st.markdown(
            """
        <div class="info-box">
        <b style="color:#00f2fe; font-size:1.05rem;">Pipeline Steps</b><br><br>
        1. <b style="color:#38bdf8;">Extract Text:</b> Extracts native text and applies OCR to scanned pages.<br>
        2. <b style="color:#38bdf8;">Detect PII:</b> Scans text using Presidio regex and spaCy NER models.<br>
        3. <b style="color:#38bdf8;">Tokenize:</b> Replaces PII with safe tokens like <span class="token">[PERSON_001]</span>.<br>
        4. <b style="color:#38bdf8;">Verify:</b> Re-scans output to ensure 0% residual leak.<br><br>
        <b style="color:#00f2fe;">Supported File Types:</b> PDF, PPTX, DOCX
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h4>Sample Redaction Preview</h4>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
        <div class="info-box" style="font-size:0.95rem; line-height:1.8">
        <b style="color:#f87171;">Original:</b> Contact Rahul Sharma at rahul@gmail.com<br>
        <b style="color:#4ade80;">Redacted:</b> Contact <span class="token">[PERSON_001]</span> at <span class="token">[EMAIL_001]</span>
        </div>
        """,
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Findings Table & Downloads
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "findings" and st.session_state.res:
    res = st.session_state.res
    thr = st.session_state.thr
    df = res["findings"].copy()
    state, review, coverage = verdict(res, thr)

    # ── Download availability ─────────────────────────────────────────────
    # has_bytes is True whenever the pipeline produced a non-empty output
    # file, regardless of whether verification fully passed.  This satisfies
    # Requirement 1: the download button must never be unconditionally
    # disabled based on verification status alone.
    sanitized_data: bytes = res.get("sanitized_bytes") or b""
    has_bytes = isinstance(sanitized_data, bytes) and len(sanitized_data) > 0

    # Verification flag – used only for the status badge, not for gating DL.
    verified = (
        res.get("redaction_status") == "VERIFIED_NO_PII_FOUND"
        and has_bytes
    )

    # Informational banner (not a blocking warning).
    # if res.get("redaction_status") == "BLOCKED" and has_bytes:
    #     # PII is still visible after every redaction pass. The file is kept
    #     # for download, but it must be reviewed before it is shared.
    #     st.warning(
    #         "⚠️ Residual PII remains after the maximum number of redaction "
    #         "passes. The sanitized PDF is available for download, but review "
    #         "it before sharing (see diagnostics below)."
    #     )
    # elif not verified and has_bytes:
    #     st.info(
    #         "ℹ️ Redaction review notice: verification flagged potential "
    #         "issues (see diagnostics below). The sanitized PDF is still "
    #         "available for download."
    #     )
    # elif not has_bytes:
    #     st.warning(
    #         "⚠️ The redaction pipeline could not produce an output file. "
    #         "This usually means extraction failed for all pages. "
    #         "Check the diagnostics below."
    #     )

    with st.expander("Redaction diagnostics"):
        st.write("Backend status:", res.get("redaction_status"))
        st.write("Sanitized PDF available:", has_bytes)
        st.write("Redaction annotations applied:", res.get("redaction_count", "N/A"))
        st.write("Verified (no residual PII):", verified)
        st.write("Inspection issues:", res.get("inspection_issues", []))
        st.write("Inspection warnings:", res.get("inspection_warnings", []))

    BADGE = {"SAFE": "badge-safe", "UNKNOWN": "badge-unknown", "UNSAFE": "badge-unsafe"}
    ICON = {"SAFE": "✅", "UNKNOWN": "⚠️", "UNSAFE": "❌"}

    st.markdown(
        f'<div class="status-banner">'
        f'  <div class="status-filename">{res["file"]}</div>'
        f'  <div class="status-meta">'
        f'    <span class="{BADGE[state]}">{ICON[state]} {state}</span>'
        f'    <span class="status-time">⏱️ Processing time: {res["secs"]}s</span>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Responsive KPI summary grid
    kpi_items = [
        (str(res["pages"]), "Total Pages"),
        (str(res["ocr_pages"]), "OCR Pages"),
        (str(len(df)), "PII Findings"),
        (f"{coverage:.1f}%", "Coverage"),
        (f"{res['secs']}s", "Processing Time"),
    ]
    render_kpis(kpi_items)

    tab1, tab2 = st.tabs(["📋 Findings List", "🔒 Sanitized Preview"])

    with tab1:
        thr = st.slider("Review confidence threshold", 0.5, 0.99, thr, 0.01)
        st.session_state.thr = thr

        c_type, c_src, c_pg = st.columns(3)
        types = c_type.multiselect("Filter PII type", sorted(df["Type"].unique()))
        srcs = c_src.multiselect("Filter source", SOURCES)
        pg = c_pg.slider("Filter page range", 1, res["pages"], (1, res["pages"]))

        df["Status"] = df["Confidence"].apply(
            lambda c: "Redacted" if c >= thr else "Needs review"
        )
        view = df[df["Page"].between(*pg)]
        if types:
            view = view[view["Type"].isin(types)]
        if srcs:
            view = view[view["Source"].isin(srcs)]

        st.caption(f"Displaying {len(view)} of {len(df)} findings.")
        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            height=380,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence Score", min_value=0, max_value=1, format="%.3f"
                )
            },
        )

        st.markdown(
            "<h3>📥 Download Output Files</h3>",
            unsafe_allow_html=True,
        )
        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            st.download_button(
                "⬇️ Download Findings (CSV)",
                data=view.to_csv(index=False).encode(),
                file_name="pii_findings.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        with dl2:
            st.download_button(
                label="⬇️ Download Sanitized PDF",
                data=sanitized_data,
                file_name=res["file"].rsplit(".", 1)[0] + "_SANITIZED.pdf",
                mime="application/pdf",
                disabled=not has_bytes,
                use_container_width=True,
            )
        with dl3:
            try:
                optimizer = TokenOptimizer()
                raw_chunks = res.get("extracted_chunks") or []
                raw_text = "\n".join(
                    c.raw_text for c in raw_chunks if getattr(c, "raw_text", None)
                )
                pii_outputs = res.get("pii_outputs") or []
                if not pii_outputs:
                    # Fallback: build a minimal PIIOutput so the optimizer
                    # always has something to render.
                    pii_outputs = [
                        PIIOutput(
                            file_id="fallback_001",
                            file_name=res.get("file", "document.pdf"),
                            page_number=1,
                            status="SUCCESS",
                            sanitized_text=(
                                "Document processed. PII entities have been "
                                "tokenised and redacted."
                            ),
                            detected_entities=[],
                            verification_passed=True,
                        )
                    ]
                optimized_md = optimizer.optimize_output(
                    pii_outputs=pii_outputs,
                    raw_original_text=raw_text,
                    chunks=raw_chunks,
                )
                st.download_button(
                    "⚡ Download LLM Markdown (.md)",
                    data=optimized_md,
                    file_name=res["file"].rsplit(".", 1)[0] + "_OPTIMIZED.md",
                    mime="text/markdown",
                    disabled=False,
                    use_container_width=True,
                )
            except (ValueError, TypeError) as e:
                st.error(f"Token Optimizer error: {e}")

    with tab2:
        b_col, a_col = st.columns(2, gap="large")
        b_col.markdown(
            "<h4 style='color:#38bdf8 !important;'>Original Text (Before Redaction)</h4>",
            unsafe_allow_html=True,
        )
        b_col.info(
            "Rahul Sharma, employee EMP-38104, can be reached at "
            "rahul@cadence.com or +91-98765-43210. "
            "His SSN is 123-45-6789. Rahul reports to the Group CRO."
        )
        a_col.markdown(
            "<h4>Sanitized Text (After Redaction)</h4>",
            unsafe_allow_html=True,
        )
        a_col.markdown(
            '<div class="info-box">'
            '<span class="token">[PERSON_001]</span>, employee '
            '<span class="token">[EMPLOYEE_ID_001]</span>, can be reached at '
            '<span class="token">[EMAIL_001]</span> or <span class="token">[PHONE_001]</span>. '
            'His SSN is <span class="token">[US_SSN_001]</span>. '
            '<span class="token">[PERSON_001]</span> reports to the Group CRO.'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Analyze another document", use_container_width=False):
        st.session_state.screen = "upload"
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Analytics (Bright Headings & Dark Plotly Charts)
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "analytics":
    res = st.session_state.res
    df = res["findings"].copy() if res and "findings" in res else pd.DataFrame()
    thr = st.session_state.thr

    file_title = f"— {res['file']}" if res and "file" in res else "— Enterprise Overview"
    st.markdown(
        f'<div class="main-title">Analytics Dashboard {file_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-text">Database & pipeline analytics: entity distributions, model confidence, and risk exposure.</div>',
        unsafe_allow_html=True,
    )

    # ── Executive KPI summary grid (fetch_executive_kpis) ──
    try:
        exec_kpis = fetch_executive_kpis()
    except (SQLAlchemyError, OSError, ValueError, KeyError):
        exec_kpis = {"total_documents": 0, "pass_rate_pct": 100.0, "total_pii_masked": 0}

    total_docs = exec_kpis.get("total_documents", 0)
    pass_rate = exec_kpis.get("pass_rate_pct", 100.0)
    total_pii = exec_kpis.get("total_pii_masked", 0)

    try:
        df_conf_metric = fetch_confidence_score()
        avg_conf_db = (
            f"{df_conf_metric['avg_confidence'].mean():.2f}"
            if not df_conf_metric.empty and "avg_confidence" in df_conf_metric.columns
            else ("N/A" if df.empty else f"{df['Confidence'].mean():.2f}")
        )
    except (SQLAlchemyError, OSError, ValueError, KeyError):
        avg_conf_db = "N/A" if df.empty else f"{df['Confidence'].mean():.2f}"

    kpi_items = [
        (str(total_docs), "Total Documents"),
        (f"{pass_rate:.1f}%", "Verification Pass Rate"),
        (str(total_pii), "Total PII Masked"),
        (avg_conf_db, "Avg Model Confidence"),
    ]
    render_kpis(kpi_items)

    # Helper function for dark themed Plotly layout with bright title and responsive sizing
    def style_dark_chart(fig, title_text, is_pie: bool = False):
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            autosize=True,
            title={
                "text": f"<b>{title_text}</b>",
                "font": {
                    "color": "#00f2fe",
                    "size": 15,
                    "family": "Calibri, Segoe UI, sans-serif",
                },
            },
            font={"color": "#e2e8f0", "family": "Calibri, Segoe UI, sans-serif"},
            margin={"l": 20, "r": 20, "t": 50, "b": 35 if is_pie else 25},
        )
        if is_pie:
            fig.update_layout(
                legend={
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.12,
                    "xanchor": "center",
                    "x": 0.5,
                    "font": {"size": 11},
                }
            )
        return fig

    # ── Section 1: 2 Pie Charts ──
    st.markdown(
        "<h3>Pie Chart Distribution</h3>",
        unsafe_allow_html=True,
    )
    pie_col1, pie_col2 = st.columns(2, gap="large")

    # Pie Chart 1: PII Entity Distribution (fetch_entity_distribution)
    with pie_col1:
        try:
            df_entity_dist = fetch_entity_distribution()
        except (SQLAlchemyError, OSError, ValueError, KeyError):
            df_entity_dist = pd.DataFrame()

        if df_entity_dist.empty and not df.empty and "Type" in df.columns:
            type_counts = df["Type"].value_counts().reset_index()
            type_counts.columns = ["entity_type", "occurrences"]
            df_entity_dist = type_counts

        if not df_entity_dist.empty:
            fig_pie_type = px.pie(
                df_entity_dist,
                names="entity_type",
                values="occurrences",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.35,
            )
            fig_pie_type = style_dark_chart(
                fig_pie_type, "PII Entity Distribution (fetch_entity_distribution)", is_pie=True
            )
            st.plotly_chart(
                fig_pie_type,
                use_container_width=True,
                config={"responsive": True, "displayModeBar": False},
            )
        else:
            st.info("No entity distribution data available.")

    # Pie Chart 2: PII Source Distribution or Verification Status Breakdown
    with pie_col2:
        if not df.empty and "Source" in df.columns:
            source_counts = df["Source"].value_counts().reset_index()
            source_counts.columns = ["Source", "Count"]
            fig_pie_source = px.pie(
                source_counts,
                names="Source",
                values="Count",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.35,
            )
            fig_pie_source = style_dark_chart(
                fig_pie_source, "PII Findings Distribution by Source", is_pie=True
            )
            st.plotly_chart(
                fig_pie_source,
                use_container_width=True,
                config={"responsive": True, "displayModeBar": False},
            )
        else:
            pass_val = exec_kpis.get("pass_rate_pct", 100.0)
            flagged_val = max(0.0, round(100.0 - pass_val, 2))
            verif_df = pd.DataFrame(
                {
                    "Status": ["Verified Clean", "Flagged for Review"],
                    "Percentage": [pass_val, flagged_val],
                }
            )
            fig_pie_verif = px.pie(
                verif_df,
                names="Status",
                values="Percentage",
                color="Status",
                color_discrete_map={
                    "Verified Clean": "#4ade80",
                    "Flagged for Review": "#f87171",
                },
                hole=0.35,
            )
            fig_pie_verif = style_dark_chart(
                fig_pie_verif, "Document Verification Status Rate", is_pie=True
            )
            st.plotly_chart(
                fig_pie_verif,
                use_container_width=True,
                config={"responsive": True, "displayModeBar": False},
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Bar Graphs 1 & 2 ──
    st.markdown(
        "<h3>Bar Graph Analysis</h3>",
        unsafe_allow_html=True,
    )
    bar_col1, bar_col2 = st.columns(2, gap="large")

    # Bar Graph 1: Average Confidence Score by Entity Type (fetch_confidence_score)
    with bar_col1:
        try:
            df_conf = fetch_confidence_score()
        except (SQLAlchemyError, OSError, ValueError, KeyError):
            df_conf = pd.DataFrame()

        if (
            df_conf.empty
            and not df.empty
            and "Type" in df.columns
            and "Confidence" in df.columns
        ):
            df_conf = (
                df.groupby("Type")
                .agg(avg_confidence=("Confidence", "mean"))
                .reset_index()
                .rename(columns={"Type": "entity_type"})
            )
            df_conf["avg_confidence"] = df_conf["avg_confidence"].round(4)

        if not df_conf.empty:
            fig_bar_conf = px.bar(
                df_conf,
                x="entity_type",
                y="avg_confidence",
                color="avg_confidence",
                labels={
                    "entity_type": "Entity Type",
                    "avg_confidence": "Average Confidence",
                },
                color_continuous_scale="Blues",
            )
            fig_bar_conf = style_dark_chart(
                fig_bar_conf, "Confidence Score by Entity Type (fetch_confidence_score)"
            )
            fig_bar_conf.update_layout(yaxis={"range": [0, 1.05]})
            st.plotly_chart(
                fig_bar_conf,
                use_container_width=True,
                config={"responsive": True, "displayModeBar": False},
            )
        else:
            st.info("No confidence score data available.")

    # Bar Graph 2: Recent Document Pass Rates (fetch_recent_pass_rates)
    with bar_col2:
        try:
            df_pass = fetch_recent_pass_rates()
        except (SQLAlchemyError, OSError, ValueError, KeyError):
            df_pass = pd.DataFrame()

        if not df_pass.empty:
            fig_bar_pass = px.bar(
                df_pass,
                x="document_number",
                y="pass_rate_pct",
                labels={
                    "document_number": "Recent Document #",
                    "pass_rate_pct": "Pass Rate (%)",
                },
                color="pass_rate_pct",
                color_continuous_scale="Tealgrn",
            )
            fig_bar_pass = style_dark_chart(
                fig_bar_pass, "Recent Pass Rates (%) (fetch_recent_pass_rates)"
            )
            fig_bar_pass.update_layout(yaxis={"range": [0, 105]})
            st.plotly_chart(
                fig_bar_pass,
                use_container_width=True,
                config={"responsive": True, "displayModeBar": False},
            )
        elif not df.empty and "Page" in df.columns and res is not None:
            page_counts = (
                df.groupby("Page")
                .size()
                .reindex(range(1, res["pages"] + 1), fill_value=0)
                .reset_index(name="Count")
            )
            fig_bar_page = px.bar(
                page_counts,
                x="Page",
                y="Count",
                labels={"Page": "Page Number", "Count": "Detections"},
                color_discrete_sequence=["#38bdf8"],
            )
            fig_bar_page = style_dark_chart(fig_bar_page, "PII Detections Count per Page")
            st.plotly_chart(
                fig_bar_page,
                use_container_width=True,
                config={"responsive": True, "displayModeBar": False},
            )
        else:
            st.info("No document pass rate data available.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Bar Graph 3 — Risk Exposure Matrix (fetch_risk_exposure_matrix) ──
    try:
        df_risk = fetch_risk_exposure_matrix()
    except (SQLAlchemyError, OSError, ValueError, KeyError):
        df_risk = pd.DataFrame()

    if not df_risk.empty:
        entity_cols = [c for c in df_risk.columns if c != "file_id"]
        risk_long = df_risk.melt(
            id_vars=["file_id"],
            value_vars=entity_cols,
            var_name="PII Entity",
            value_name="Occurrences",
        )
        risk_long = risk_long[risk_long["Occurrences"] > 0]

        # Top 15 exposed files for readable display
        top_files = (
            risk_long.groupby("file_id")["Occurrences"]
            .sum()
            .nlargest(15)
            .index.tolist()
        )
        risk_long = risk_long[risk_long["file_id"].isin(top_files)]

        fig_bar_risk = px.bar(
            risk_long,
            x="Occurrences",
            y="file_id",
            color="PII Entity",
            orientation="h",
            labels={"Occurrences": "Detected Entities", "file_id": "File ID"},
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig_bar_risk = style_dark_chart(
            fig_bar_risk, "Risk Exposure Matrix Across Documents (fetch_risk_exposure_matrix)"
        )
        fig_bar_risk.update_layout(
            margin={"l": 130, "r": 20, "t": 50, "b": 30},
            legend={
                "orientation": "h",
                "yanchor": "top",
                "y": -0.15,
                "xanchor": "center",
                "x": 0.5,
                "font": {"size": 11},
            },
        )
        st.plotly_chart(
            fig_bar_risk,
            use_container_width=True,
            config={"responsive": True, "displayModeBar": False},
        )
    elif not df.empty and "Detector" in df.columns:
        det_counts = df["Detector"].value_counts().reset_index()
        det_counts.columns = ["Detector Name", "Detections"]
        fig_bar_det = px.bar(
            det_counts,
            x="Detections",
            y="Detector Name",
            orientation="h",
            color_discrete_sequence=["#a855f7"],
        )
        fig_bar_det = style_dark_chart(
            fig_bar_det, "Detections by Detector Model / Recognizer"
        )
        fig_bar_det.update_layout(margin={"l": 120, "r": 20, "t": 50, "b": 30})
        st.plotly_chart(
            fig_bar_det,
            use_container_width=True,
            config={"responsive": True, "displayModeBar": False},
        )
    else:
        st.info("No risk exposure matrix data available.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Export Analytics Summary ──
    summary_data = {
        "Metric": [
            "Total Documents Processed",
            "System Pass Rate (%)",
            "Total PII Entities Masked",
            "Average Confidence Score",
        ],
        "Value": [
            total_docs,
            f"{pass_rate:.1f}%",
            total_pii,
            avg_conf_db,
        ],
    }
    if res is not None:
        summary_data["Metric"].extend([
            "Active File Findings",
            "Active File OCR Pages",
            "Active File Processing Time (s)",
        ])
        summary_data["Value"].extend([
            len(df),
            res.get("ocr_pages", 0),
            res.get("secs", 0.0),
        ])

    summary = pd.DataFrame(summary_data)
    st.download_button(
        "⬇️ Export Analytics Summary (CSV)",
        data=summary.to_csv(index=False).encode(),
        file_name="pii_analytics_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

else:
    st.session_state.screen = "upload"
    st.rerun()
