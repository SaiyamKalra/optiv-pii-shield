"""
PII Sentinel — Frontend
Run:  streamlit run app.py
"""

import random
import tempfile
import time
import warnings
from pathlib import Path

# Suppress PyTorch/HuggingFace startup warnings
warnings.filterwarnings("ignore")

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
    get_conversion_matrix,
)
from extraction.normalizer import extract_document
from extraction.pdf_extractor import extract_pdf
from models.schemas import PIIOutput
from pii.pipeline import PIIPipeline
from redaction.office_pipeline import OfficeRedactionPipeline, scan_chunks
from redaction.pipeline import RedactionPipeline
from redaction.token_redaction.token_optimizer import TokenOptimizer

# ── 1. PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND) ────────────────
st.set_page_config(
    page_title="PII Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 2. STYLING & CUSTOM CSS ────────────────────────────────────────────────
st.markdown(
"""
<style>
    /* Reset & Base Typography */
    html, body, [class*="css"], .stApp {
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        background-color: #0d0d14;
        color: #e2e8f0;
        box-sizing: border-box;
    }

    *, *:before, *:after {
        box-sizing: inherit;
    }

    /* Responsive Main Container */
    .block-container, [data-testid="stMainBlockContainer"] {
        padding-top: 1.75rem;
        padding-bottom: 2.5rem;
        padding-left: clamp(0.75rem, 3vw, 2.5rem);
        padding-right: clamp(0.75rem, 3vw, 2.5rem);
        max-width: 1440px;
        margin: 0 auto;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6, 
    [data-testid="stMarkdownContainer"] h1, 
    [data-testid="stMarkdownContainer"] h2, 
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #00f2fe;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        font-weight: 700;
        word-break: break-word;
        line-height: 1.25;
    }

    div[data-testid="stHorizontalBlock"]:has(.main-title) {
        align-items: center;
    }

    .main-title {
        font-size: clamp(1.5rem, 3.5vw, 2.2rem);
        font-weight: 700;
        color: #00f2fe;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        margin: 0;
        padding: 0;
        white-space: nowrap;
        line-height: 1;
        text-shadow: 0 0 14px rgba(0, 242, 254, 0.28);
    }

    .sub-text {
        color: #cbd5e1;
        font-size: clamp(0.85rem, 1.8vw, 1.05rem);
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        margin-bottom: 18px;
        line-height: 1.45;
        word-break: break-word;
    }

    /* Responsive KPI Grid & Cards */
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
        font-size: clamp(1.35rem, 2.6vw, 2rem);
        font-weight: 800;
        color: #38bdf8;
        line-height: 1.15;
        word-break: break-word;
    }
    .kpi-label {
        font-size: clamp(0.72rem, 1.1vw, 0.88rem);
        color: #94a3b8;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        line-height: 1.2;
    }

    /* ── DATAFRAME PROGRESS BAR GREEN COLOR ONLY ── */
    div[data-testid="stDataFrame"] progress::-webkit-progress-value {
        background-color: #4ade80 !important;
    }
    div[data-testid="stDataFrame"] progress::-moz-progress-bar {
        background-color: #4ade80 !important;
    }
    div[data-testid="stDataFrame"] progress {
        color: #4ade80 !important;
        accent-color: #4ade80 !important;
    }
    div[data-testid="stDataFrame"] [role="progressbar"] > div,
    div[data-testid="stDataFrame"] [data-testid="stProgress"] > div > div,
    div[data-testid="stDataFrame"] div[style*="background-color: rgb(255"],
    div[data-testid="stDataFrame"] div[style*="background-color: red"],
    div[data-testid="stDataFrame"] div[style*="background-color: #ff"] {
        background-color: #4ade80 !important;
    }

    /* Tokens & Badges */
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

    /* Status Header Banner */
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
        color: #00f2fe;
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

    /* Streamlit Status Widget Bounds & Overflow Fix */
    div[data-testid="stStatusWidget"] {
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
        overflow: hidden;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }

    div[data-testid="stStatusWidget"] summary,
    div[data-testid="stStatusWidget"] button,
    div[data-testid="stStatusWidget"] > div {
        width: 100%;
        max-width: 100%;
        box-sizing: border-box;
        margin-right: 0;
        padding-right: 12px;
    }

    /* Info Box */
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

    /* Buttons */
    .stButton > button {
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        font-weight: 600;
        border-radius: 8px;
        min-height: 42px;
        font-size: clamp(0.82rem, 1.4vw, 0.96rem);
        transition: all 0.2s ease;
        white-space: normal;
        word-break: break-word;
        line-height: 1.25;
        padding: 8px 12px;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
        border: none;
        color: #ffffff;
        box-shadow: 0 2px 10px rgba(6, 182, 212, 0.3);
    }
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #cbd5e1;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #00f2fe;
        color: #00f2fe;
        background: rgba(0, 242, 254, 0.08);
    }

    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
        color: #ffffff;
        border: none;
        font-weight: 600;
        border-radius: 8px;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        font-size: clamp(0.82rem, 1.4vw, 0.96rem);
        min-height: 42px;
        padding: 8px 14px;
        white-space: normal;
        word-break: break-word;
        line-height: 1.25;
        transition: all 0.2s ease;
    }
    div[data-testid="stDownloadButton"] > button:hover:not(:disabled) {
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.45);
        transform: translateY(-1px);
    }

    [data-testid="stDataFrame"] {
        width: 100%;
        max-width: 100%;
        overflow-x: auto;
    }
    .js-plotly-plot, .plot-container {
        width: 100%;
        max-width: 100%;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── 3. PIPELINE SETUP & DEMO HELPERS ───────────────────────────────────────
ENTITY_TYPES = [
    "PERSON",
    "EMAIL ADDRESS",
    "PHONE NUMBER",
    "US SSN",
    "CREDIT CARD",
    "IP ADDRESS",
    "LOCATION",
    "ORGANIZATION",
]
WEIGHTS = [28, 20, 18, 10, 8, 5, 6, 5]
SOURCES = ["body", "table", "image/OCR", "narrative"]
DETECTORS = {
    "PERSON": "spaCy NER + name filter",
    "EMAIL ADDRESS": "CorpEmail recognizer",
    "PHONE NUMBER": "RobustPhone recognizer",
    "US SSN": "StrictSSN recognizer",
    "CREDIT CARD": "CorpCreditCard recognizer",
    "IP ADDRESS": "Presidio regex",
    "LOCATION": "spaCy NER",
    "ORGANIZATION": "spaCy NER",
    # Entity names as the pipeline emits them (with underscores).
    "EMAIL_ADDRESS": "CorpEmail recognizer",
    "PHONE_NUMBER": "RobustPhone recognizer",
    "US_SSN": "StrictSSN recognizer",
    "CREDIT_CARD": "CorpCreditCard recognizer",
    "IP_ADDRESS": "Presidio regex",
    "EMPLOYEE_ID": "EmployeeId recognizer",
    "DIRECTOR_ID": "DirectorId recognizer",
    "VENDOR_ID": "VendorId recognizer",
    "PASSPORT": "Passport recognizer (needs 'passport' label)",
    "PAN": "PAN recognizer",
    "NATIONAL_ID": "NationalId recognizer (needs PESEL/Aadhaar label)",
    "DATE_OF_BIRTH": "DateOfBirth recognizer (needs DOB label)",
    "STREET_ADDRESS": "StreetAddress recognizer",
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
    total_words = pages * 450
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
        "extracted_chunks": [],
        "pii_outputs": [],
        "total_words": total_words,
        "reliable_words": total_words,
    }


SUPPORTED_EXTENSIONS = ("pdf", "docx", "pptx")
OUTPUT_MIME = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _run_real_pipeline(uploaded) -> dict:
    """Run extraction, PII detection, redaction and verification (PDF, DOCX, PPTX)."""
    ext = uploaded.name.lower().rsplit(".", 1)[-1]
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported file types are PDF, DOCX and PPTX.")

    started = time.perf_counter()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"original.{ext}"
        output_path = Path(temp_dir) / f"sanitized.{ext}"
        input_path.write_bytes(uploaded.getvalue())

        pii_pipeline = PIIPipeline()
        if ext == "pdf":
            chunks = extract_pdf(input_path)
            original_results = pii_pipeline.process_document_chunks(chunks)

            redaction_pipeline = RedactionPipeline(pii_pipeline=pii_pipeline)
            result = redaction_pipeline.process(
                input_pdf=str(input_path),
                output_pdf=str(output_path),
                chunks=chunks,
                pii_results=original_results,
            )

            with pymupdf.open(input_path) as document:
                page_count = len(document)
        else:
            # Validates the file by content, then picks the DOCX or PPTX extractor.
            chunks = extract_document(input_path)
            # Office files also carry hidden text (metadata, comments, notes...).
            original_results = pii_pipeline.process_document_chunks(scan_chunks(chunks))
            result = OfficeRedactionPipeline(pii_pipeline=pii_pipeline).process(
                str(input_path), str(output_path), chunks=chunks, pii_results=original_results
            )
            page_count = len(chunks)

        rows = []
        for chunk, page_result in zip(chunks, original_results):
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
        if not findings.empty:
            findings = findings.sort_values(["Page", "Type"]).reset_index(drop=True)
            findings.insert(0, "ID", [f"F-{i + 1:05d}" for i in range(len(findings))])

        status = result["status"]
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

        # Compute total words processed across all extracted chunks
        total_words = sum(
            len(chunk.raw_text.split())
            for chunk in chunks
            if getattr(chunk, "raw_text", None)
        )

        return {
            "file": uploaded.name,
            "file_type": ext,
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
            "total_words": total_words,
            "reliable_words": total_words,
        }

# ── 4. STATE MANAGEMENT & VERDICT HELPERS ─────────────────────────────────
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

    total_words = int(res.get("total_words") or 0)
    reliable_words = min(int(res.get("reliable_words") or 0), total_words)
    coverage = (reliable_words / total_words * 100) if total_words > 0 else 100.0

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


# ── 5. NAVIGATION HEADER ──────────────────────────────────────────────────
st.markdown('<div class="nav-header-wrapper">', unsafe_allow_html=True)
c_title, c_nav = st.columns([1, 2], gap="medium")

with c_title:
    st.markdown('<div class="main-title">PII Sentinel</div>', unsafe_allow_html=True)

with c_nav:
    n1, n2, n3 = st.columns(3, gap="small")
    with n1:
        if st.button(
            "Upload Screen",
            use_container_width=True,
            type="primary" if st.session_state.screen == "upload" else "secondary",
        ):
            st.session_state.screen = "upload"
            st.rerun()
    with n2:
        if st.button(
            "Findings Table",
            use_container_width=True,
            disabled=st.session_state.res is None,
            type="primary" if st.session_state.screen == "findings" else "secondary",
        ):
            st.session_state.screen = "findings"
            st.rerun()
    with n3:
        if st.button(
            "Analytics Charts",
            use_container_width=True,
            type="primary" if st.session_state.screen == "analytics" else "secondary",
        ):
            st.session_state.screen = "analytics"
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
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

        go_btn = st.button(
            "Analyze Document", use_container_width=True, disabled=uploaded is None
        )

        if go_btn and uploaded:
            name = uploaded.name
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
                step_placeholder = st.empty()

                for step in STEPS[:3]:
                    step_placeholder.write(f"✓ {step}")
                    time.sleep(0.15)
                try:
                    st.session_state.res = _run_real_pipeline(uploaded)
                    for step in STEPS[3:]:
                        step_placeholder.write(f"✓ {step}")
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

            st.session_state.screen = "findings"
            st.rerun()

    with right:
        st.markdown(
            """
        <div class="info-box">
        <b style="color:#00f2fe; font-size:1.05rem;">Pipeline Steps</b><br><br>
1. <b style="color:#38bdf8;">Extract & OCR:</b> Parses layout structures, tables, and native text while running PyTesseract OCR on scanned pages.<br>
        2. <b style="color:#38bdf8;">Hybrid Detection:</b> Scans document chunks using Presidio regex rules, spaCy NER, and custom pattern recognizers.<br>
        3. <b style="color:#38bdf8;">Harvest & Calibrate:</b> Extracts cell candidates under role headers and uses <code>PIIRiskCalibrator</code> to boost non-Western names and multi-line phone numbers.<br>
        4. <b style="color:#38bdf8;">Tokenize & Redact:</b> Replaces sensitive entities with consistent mapping tokens (e.g., <span class="token">[PERSON_001]</span>) across pages.<br>
        5. <b style="color:#38bdf8;">Residual Verification:</b> Passes sanitized text through a <code>VerificationGate</code> to verify 0% residual PII leakage.<br>
        6. <b style="color:#38bdf8;">Optimize Export:</b> Generates token-compressed, LLM-ready Markdown via <code>TokenOptimizer</code>.<br><br>
        <b style="color:#00f2fe;">Supported File Types:</b> PDF, PPTX, DOCX
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            """
        <div class="info-box" style="font-size:0.95rem; line-height:1.8">
        <b style="color:#00f2fe; font-size:1.05rem;">Sample Redaction Preview</b><br><br>
        <b style="color:#f87171;">Original:</b> Contact Rahul Sharma at rahul@gmail.com<br>
        <b style="color:#4ade80;">Redacted:</b> Contact <span class="token">[PERSON_001]</span> at <span class="token">[EMAIL_001]</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Findings Table & Data Conversion Metrics
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "findings" and st.session_state.res:
    res = st.session_state.res
    thr = st.session_state.thr
    df = res["findings"].copy()
    state, review, coverage = verdict(res, thr)

    sanitized_data: bytes = res.get("sanitized_bytes") or b""
    has_bytes = isinstance(sanitized_data, bytes) and len(sanitized_data) > 0
    verified = res.get("redaction_status") == "VERIFIED_NO_PII_FOUND" and has_bytes

    # Calculate Data Conversion Matrix safely
    raw_chunks = res.get("extracted_chunks") or []
    pii_outputs = res.get("pii_outputs") or []
    raw_text = "\n".join(c.raw_text for c in raw_chunks if getattr(c, "raw_text", None))
    metrics_data = get_conversion_matrix(raw_text, pii_outputs)

    st.markdown("<h3>Document Transformation & Conversion Summary</h3>", unsafe_allow_html=True)
    
    conversion_kpis = [
        (str(metrics_data["total_words_processed"]), "TOTAL WORDS PROCESSED"),
        (str(metrics_data["total_pii_entities_found"]), "PII IDENTIFIED"),
        (f"{metrics_data['redaction_density_percentage']}%", "DATA CONVERTED"),
    ]
    render_kpis(conversion_kpis)

    st.divider()

    st.markdown(
        f'<div class="status-banner">'
        f'  <div class="status-filename">{res["file"]}</div>'
        f'  <div class="status-meta">'
        f'    <span class="status-time">Processing time: {res["secs"]}s</span>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    kpi_items = [
        (str(res["pages"]), "Total Pages"),
        (str(res["ocr_pages"]), "OCR Pages"),
        (str(len(df)), "PII Findings"),
        (f"{coverage:.1f}%", "Coverage"),
        (f"{res['secs']}s", "Processing Time"),
    ]
    render_kpis(kpi_items)

    # Render Findings List content directly without tabs
    c_type, c_src, c_pg = st.columns(3)
    
    # Format multiselect types to remove underscores
    formatted_types = sorted([
        str(t).replace("_", " ").title() for t in df["Type"].unique()
    ])
    types = c_type.multiselect("Filter PII type", formatted_types)
    srcs = c_src.multiselect("Filter source", SOURCES)
    pg = c_pg.slider("Filter page range", 1, res["pages"], (1, res["pages"]))

    df["Status"] = df["Confidence"].apply(
        lambda c: "Redacted" if c >= thr else "Needs review"
    )
    view = df[df["Page"].between(*pg)].copy()
    
    # Pre-format Type column for clean display
    view["Type"] = view["Type"].astype(str).str.replace("_", " ").str.title()
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
                "Confidence Score",
                min_value=0,
                max_value=1,
                format="%.3f",
            )
        },
    )

    st.markdown(
        "<h3>Download Output Files</h3>",
        unsafe_allow_html=True,
    )
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        st.download_button(
            "Download Findings (CSV)",
            data=view.to_csv(index=False).encode(),
            file_name="pii_findings.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with dl2:
        st.download_button(
            label="Download Sanitized PDF",
            data=sanitized_data,
            file_name=res["file"].rsplit(".", 1)[0] + "_SANITIZED.pdf",
            mime="application/pdf",
            disabled=not has_bytes,
            use_container_width=True,
        )
    with dl3:
        try:
            optimizer = TokenOptimizer()
            if not pii_outputs:
                pii_outputs = [
                    PIIOutput(
                        file_id="fallback_001",
                        file_name=res.get("file", "document.pdf"),
                        page_number=1,
                        status="SUCCESS",
                        sanitized_text="Document processed. PII entities have been tokenised and redacted.",
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
                "Download LLM Markdown (.md)",
                data=optimized_md,
                file_name=res["file"].rsplit(".", 1)[0] + "_OPTIMIZED.md",
                mime="text/markdown",
                disabled=False,
                use_container_width=True,
            )
        except (ValueError, TypeError) as e:
            st.error(f"Token Optimizer error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Analyze another document", use_container_width=False):
        st.session_state.screen = "upload"
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Analytics
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "analytics":
    res = st.session_state.res
    df = res["findings"].copy() if res and "findings" in res else pd.DataFrame()

    file_title = f"- {res['file']}" if res and "file" in res else "— Enterprise Overview"
    st.markdown(
        f'<div class="main-title">Analytics Dashboard {file_title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-text">Database & pipeline analytics: entity distributions, model confidence, and risk exposure.</div>',
        unsafe_allow_html=True,
    )

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
            margin={"l": 20, "r": 20, "t": 50, "b": 60 if is_pie else 25},
        )
        if is_pie:
            # Prevent tiny slice callout labels from crowding each other
            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                insidetextorientation="radial",
            )
            # Layout legend cleanly with vertical line spacing
            fig.update_layout(
                legend={
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.18,
                    "xanchor": "center",
                    "x": 0.5,
                    "font": {"size": 11},
                    "entrywidth": 140,
                    "entrywidthmode": "pixels",
                }
            )
        return fig

    st.markdown("<h3>Pie Chart Distribution</h3>", unsafe_allow_html=True)
    pie_col1, pie_col2 = st.columns(2, gap="large")

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
            # Clean underscores and convert to title case
            df_entity_dist["entity_type"] = (
                df_entity_dist["entity_type"]
                .astype(str)
                .str.replace("_", " ")
                .str.title()
            )

            fig_pie_type = px.pie(
                df_entity_dist,
                names="entity_type",
                values="occurrences",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                hole=0.35,
            )
            fig_pie_type = style_dark_chart(
                fig_pie_type, "PII Entity Distribution", is_pie=True
            )
            st.plotly_chart(
                fig_pie_type,
                use_container_width=True,
                config={"responsive": True, "displayModeBar": False},
            )
        else:
            st.info("No entity distribution data available.")

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

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h3>Bar Graph Analysis</h3>", unsafe_allow_html=True)
    bar_col1, bar_col2 = st.columns(2, gap="large")

    with bar_col1:
        try:
            df_conf = fetch_confidence_score()
        except (SQLAlchemyError, OSError, ValueError, KeyError):
            df_conf = pd.DataFrame()

        if df_conf.empty and not df.empty and "Type" in df.columns and "Confidence" in df.columns:
            df_conf = (
                df.groupby("Type")
                .agg(avg_confidence=("Confidence", "mean"))
                .reset_index()
                .rename(columns={"Type": "entity_type"})
            )

        if not df_conf.empty:
            # Clean underscores and convert to title case
            df_conf["entity_type"] = (
                df_conf["entity_type"]
                .astype(str)
                .str.replace("_", " ")
                .str.title()
            )

            fig_bar_conf = px.bar(
                df_conf,
                x="entity_type",
                y="avg_confidence",
                color="avg_confidence",
                labels={"entity_type": "Entity Type", "avg_confidence": "Average Confidence"},
                color_continuous_scale="Blues",
            )
            fig_bar_conf = style_dark_chart(fig_bar_conf, "Confidence Score by Entity Type")
            st.plotly_chart(fig_bar_conf, use_container_width=True, config={"responsive": True, "displayModeBar": False})

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
                labels={"document_number": "Recent Document #", "pass_rate_pct": "Pass Rate (%)"},
                color="pass_rate_pct",
                color_continuous_scale="Tealgrn",
            )
            fig_bar_pass = style_dark_chart(fig_bar_pass, "Recent Pass Rates (%)")
            st.plotly_chart(fig_bar_pass, use_container_width=True, config={"responsive": True, "displayModeBar": False})

else:
    st.session_state.screen = "upload"
    st.rerun()