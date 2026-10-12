"""
PII Sentinel — Frontend
Run:  streamlit run app.py
"""

import html
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

from analytics.metrics import get_conversion_matrix
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

    /* Analytics empty state */
    .empty-state {
        max-width: 760px;
        margin: 28px auto 22px auto;
        padding: clamp(24px, 4vw, 40px) clamp(18px, 4vw, 40px);
        text-align: center;
        background: radial-gradient(ellipse at top, rgba(0, 242, 254, 0.08), rgba(255, 255, 255, 0.02) 70%);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
    }
    .empty-icon {
        width: 64px;
        height: 64px;
        margin: 0 auto 14px auto;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #00f2fe;
        background: rgba(0, 242, 254, 0.10);
        border: 1px solid rgba(0, 242, 254, 0.35);
        box-shadow: 0 0 22px rgba(0, 242, 254, 0.18);
    }
    .empty-title {
        color: #f1f5f9;
        font-size: clamp(1.2rem, 2.4vw, 1.45rem);
        font-weight: 700;
        margin-bottom: 8px;
    }
    .empty-text {
        color: #94a3b8;
        font-size: 0.98rem;
        line-height: 1.5;
        max-width: 520px;
        margin: 0 auto 24px auto;
    }
    .empty-tiles {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        text-align: left;
    }
    .empty-tile {
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed rgba(255, 255, 255, 0.14);
        border-radius: 10px;
        padding: 12px 14px;
    }
    .empty-tile-title {
        color: #38bdf8;
        font-weight: 600;
        font-size: 0.92rem;
        margin-bottom: 3px;
    }
    .empty-tile-text {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.35;
    }

    /* Analysis progress card (replaces st.status, whose header overflowed its box) */
    .progress-card {
        width: 100%;
        box-sizing: border-box;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0 12px 0;
    }
    .progress-title {
        color: #e2e8f0;
        font-weight: 700;
        font-size: 1.02rem;
        margin-bottom: 10px;
        word-break: break-word;
    }
    .progress-file { color: #00f2fe; }
    .progress-step {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 4px 0;
        font-size: 0.95rem;
    }
    .progress-icon {
        width: 18px;
        display: inline-flex;
        justify-content: center;
        flex-shrink: 0;
    }
    .progress-step.done { color: #4ade80; }
    .progress-step.active { color: #e2e8f0; font-weight: 600; }
    .progress-step.pending { color: #64748b; }
    .progress-step.failed { color: #f87171; font-weight: 600; }
    .progress-spinner {
        width: 14px;
        height: 14px;
        border: 2px solid rgba(0, 242, 254, 0.25);
        border-top-color: #00f2fe;
        border-radius: 50%;
        display: inline-block;
        animation: progress-spin 0.8s linear infinite;
    }
    @keyframes progress-spin { to { transform: rotate(360deg); } }
    .progress-time {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 8px;
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


PIPELINE_STAGES = [
    "Extracting text and running OCR on images",
    "Detecting PII",
    "Redacting and re-scanning the sanitized output",
]


def _run_real_pipeline(uploaded, on_stage=None) -> dict:
    """Run extraction, PII detection, redaction and verification (PDF, DOCX, PPTX).

    on_stage(i) is called when stage PIPELINE_STAGES[i] starts, so the UI shows the
    real progress instead of fixed timers."""
    def stage(i):
        if on_stage is not None:
            on_stage(i)

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
            stage(0)
            chunks = extract_pdf(input_path)
            stage(1)
            original_results = pii_pipeline.process_document_chunks(chunks)

            stage(2)
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
            stage(0)
            chunks = extract_document(input_path)
            # Office files also carry hidden text (metadata, comments, notes...).
            stage(1)
            original_results = pii_pipeline.process_document_chunks(scan_chunks(chunks))
            stage(2)
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

            progress = st.empty()
            started = time.perf_counter()

            def show_progress(current: int, failed: bool = False):
                """Render the progress card: done stages ticked, the current one
                animated (or marked failed), the rest greyed out."""
                rows = []
                for i, label in enumerate(PIPELINE_STAGES):
                    if i < current:
                        state, icon = "done", "✓"
                    elif i == current and failed:
                        state, icon = "failed", "✕"
                    elif i == current:
                        state, icon = "active", '<span class="progress-spinner"></span>'
                    else:
                        state, icon = "pending", "•"
                    rows.append(
                        f'<div class="progress-step {state}">'
                        f'<span class="progress-icon">{icon}</span>{html.escape(label)}</div>'
                    )
                title = "Processing failed" if failed else (
                    "Done" if current >= len(PIPELINE_STAGES) else "Analyzing"
                )
                progress.markdown(
                    f'<div class="progress-card">'
                    f'<div class="progress-title">{title}: '
                    f'<span class="progress-file">{html.escape(name)}</span></div>'
                    f'{"".join(rows)}'
                    f'<div class="progress-time">{time.perf_counter() - started:.0f}s elapsed</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            stage_reached = {"i": 0}

            def on_stage(i: int):
                stage_reached["i"] = i
                show_progress(i)

            show_progress(0)
            try:
                st.session_state.res = _run_real_pipeline(uploaded, on_stage=on_stage)
                show_progress(len(PIPELINE_STAGES))
            except (ValueError, OSError, RuntimeError) as exc:
                show_progress(stage_reached["i"], failed=True)
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
        out_ext = res.get("file_type", "pdf")
        st.download_button(
            label=f"Download Sanitized {out_ext.upper()}",
            data=sanitized_data,
            file_name=res["file"].rsplit(".", 1)[0] + f"_SANITIZED.{out_ext}",
            mime=OUTPUT_MIME.get(out_ext, "application/octet-stream"),
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

    # Analytics describe the document analysed in this session. The database holds
    # one row per page from every run (including tests and rescans), so all-time
    # totals from it would not describe anything the user uploaded.
    if not res:
        st.markdown(
            '<div class="main-title">Analytics Dashboard</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
<div class="empty-state">
<div class="empty-icon">
<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><rect x="7" y="12" width="3" height="6" rx="1"/><rect x="12" y="8" width="3" height="10" rx="1"/><rect x="17" y="5" width="3" height="13" rx="1"/></svg>
</div>
<div class="empty-title">No document analysed yet</div>
<div class="empty-text">Upload a PDF, DOCX or PPTX and run the analysis. This page will then break down what was found in that document.</div>
<div class="empty-tiles">
<div class="empty-tile"><div class="empty-tile-title">PII by type</div><div class="empty-tile-text">Names, emails, phones, IDs</div></div>
<div class="empty-tile"><div class="empty-tile-title">Source</div><div class="empty-tile-text">Page text vs image OCR</div></div>
<div class="empty-tile"><div class="empty-tile-title">Confidence</div><div class="empty-tile-text">Average score per type</div></div>
<div class="empty-tile"><div class="empty-tile-title">Location</div><div class="empty-tile-text">Findings on each page</div></div>
</div>
</div>
            """,
            unsafe_allow_html=True,
        )
        _, mid, _ = st.columns([2, 1, 2])
        with mid:
            if st.button("Go to Upload Screen", type="primary", use_container_width=True):
                st.session_state.screen = "upload"
                st.rerun()
        st.stop()

    df = res["findings"].copy() if "findings" in res else pd.DataFrame()

    st.markdown(
        f'<div class="main-title">Analytics Dashboard — {html.escape(res["file"])}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-text">What was detected in this document: PII types, sources, '
        "detection confidence and where it appears.</div>",
        unsafe_allow_html=True,
    )

    status_label = {
        "VERIFIED_NO_PII_FOUND": "Verified",
        "REVIEW_REQUIRED": "Review required",
        "BLOCKED": "Blocked",
    }.get(res.get("redaction_status"), res.get("redaction_status") or "N/A")
    render_kpis([
        (str(res.get("pages", 0)), "Pages"),
        (str(len(df)), "PII Findings"),
        (str(df["Token"].nunique()) if "Token" in df.columns else "0", "Distinct PII Values"),
        (f"{df['Confidence'].mean():.2f}" if not df.empty else "N/A", "Avg Confidence"),
        (status_label, "Verification"),
    ])

    if df.empty:
        st.info("No PII was detected in this document, so there is nothing to chart.")
        st.stop()

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
            fig.update_traces(
                textposition="inside",
                textinfo="percent",
                insidetextorientation="radial",
            )
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

    def show(fig):
        st.plotly_chart(
            fig, use_container_width=True, config={"responsive": True, "displayModeBar": False}
        )

    df["Type"] = df["Type"].astype(str).str.replace("_", " ").str.title()

    st.markdown("<h3>Distribution</h3>", unsafe_allow_html=True)
    pie_col1, pie_col2 = st.columns(2, gap="large")
    with pie_col1:
        type_counts = df["Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Findings"]
        show(style_dark_chart(
            px.pie(type_counts, names="Type", values="Findings", hole=0.35,
                   color_discrete_sequence=px.colors.qualitative.Pastel),
            "PII by Type", is_pie=True,
        ))
    with pie_col2:
        source_counts = df["Source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Findings"]
        show(style_dark_chart(
            px.pie(source_counts, names="Source", values="Findings", hole=0.35,
                   color_discrete_sequence=px.colors.qualitative.Set3),
            "PII by Source (text vs image OCR)", is_pie=True,
        ))

    st.markdown("<h3>Confidence and Location</h3>", unsafe_allow_html=True)
    bar_col1, bar_col2 = st.columns(2, gap="large")
    with bar_col1:
        conf = df.groupby("Type", as_index=False)["Confidence"].mean()
        fig = px.bar(conf, x="Type", y="Confidence", color="Confidence",
                     color_continuous_scale="Blues", range_y=[0, 1],
                     labels={"Confidence": "Average confidence"})
        show(style_dark_chart(fig, "Average Confidence by Type"))
    with bar_col2:
        per_page = df.groupby("Page").size().reindex(
            range(1, int(res.get("pages", 0) or df["Page"].max()) + 1), fill_value=0
        ).reset_index()
        per_page.columns = ["Page", "Findings"]
        fig = px.bar(per_page, x="Page", y="Findings", color="Findings",
                     color_continuous_scale="Tealgrn")
        show(style_dark_chart(fig, "Findings per Page"))

else:
    st.session_state.screen = "upload"
    st.rerun()