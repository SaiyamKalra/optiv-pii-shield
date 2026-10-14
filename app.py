"""
PII Sentinel — Frontend
Run:  streamlit run app.py
"""

import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path

# Suppress PyTorch/HuggingFace startup warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.express as px
import pymupdf
import streamlit as st

from analytics.metrics import get_conversion_matrix, reading_coverage
from extraction.normalizer import extract_document
from extraction.pdf_extractor import extract_pdf
from pii.pipeline import PIIPipeline
from redaction.office_pipeline import OfficeRedactionPipeline, scan_chunks
from redaction.pipeline import RedactionPipeline
from redaction.token_redaction.token_optimizer import TokenOptimizer
from ui import components as ui
from ui.evidence import attention, build_trail
from ui.page_view import page_overlay, page_uncertainty, rasterize

# ── 1. PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND) ────────────────
st.set_page_config(
    page_title="PII Sentinel",
    page_icon="▮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

THEME_CSS = Path(__file__).with_name("ui") / "theme.css"
st.markdown(f"<style>{THEME_CSS.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ── 2. PIPELINE ────────────────────────────────────────────────────────────
DETECTORS = {
    "PERSON": "spaCy NER + name filter",
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

CHART_COLORS = ["#1b1b1b", "#b3261e", "#2e6b3a", "#b7791f", "#1f5f8b", "#6b4c9a", "#8c2f39", "#5b6b73"]
VIEWER_DPI = 110


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
    original_bytes = uploaded.getvalue()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = Path(temp_dir) / f"original.{ext}"
        output_path = Path(temp_dir) / f"sanitized.{ext}"
        input_path.write_bytes(original_bytes)

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
            columns=["Page", "Source", "Type", "Detector", "Confidence", "Token", "Context"],
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

        total_words = sum(
            len(chunk.raw_text.split())
            for chunk in chunks
            if getattr(chunk, "raw_text", None)
        )
        coverage = reading_coverage(chunks)

        return {
            "file": uploaded.name,
            "file_type": ext,
            "pages": page_count,
            "ocr_pages": sum(
                1
                for chunk in chunks
                if chunk.ocr_used or chunk.extraction_method in ("ocr", "hybrid")
            ),
            "findings": findings,
            "original_bytes": original_bytes,
            "sanitized_bytes": sanitized_bytes,
            "residual": (
                0 if status == "VERIFIED_NO_PII_FOUND"
                else len(result.get("inspection_issues", []))
            ),
            "secs": round(time.perf_counter() - started, 2),
            "redaction_status": status,
            "redaction_count": result.get("redaction_count", 0),
            "verification_passed": result.get("verification_passed"),
            "unmapped_findings": result.get("unmapped_findings", []),
            "inspection_issues": result.get("inspection_issues", []),
            "inspection_warnings": result.get("inspection_warnings", []),
            "extracted_chunks": chunks,
            "pii_outputs": original_results,
            "total_words": total_words,
            # Words and unread regions OCR/extraction could not vouch for count against it.
            "reliable_words": coverage["reliable"],
            "coverage_total": coverage["total"],
            "coverage": coverage,
        }


# ── 3. STATE & VERDICT ─────────────────────────────────────────────────────
for key, default in (("res", None), ("screen", "intake"), ("thr", 0.70), ("viewer_page", 1)):
    if key not in st.session_state:
        st.session_state[key] = default

COVERAGE_REQUIRED_PCT = 95.0


def verdict(res, thr):
    """Return (state, low_confidence_count, word_coverage_pct)."""
    df = res.get("findings")
    review = 0
    if df is not None and "Confidence" in df.columns:
        review = int((df["Confidence"] < thr).sum())

    total_items = int(res.get("coverage_total") or res.get("total_words") or 0)
    reliable_items = min(int(res.get("reliable_words") or 0), total_items)
    coverage = (reliable_items / total_items * 100) if total_items > 0 else 100.0

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


@st.cache_data(show_spinner=False, max_entries=48)
def _page_image(pdf_bytes: bytes, page_number: int, dpi: int):
    return rasterize(pdf_bytes, page_number, dpi)


@st.cache_resource(show_spinner=False)
def _token_optimizer() -> TokenOptimizer:
    return TokenOptimizer()


def _llm_markdown(res: dict) -> str:
    """Built once per document (the optimizer loads a language model)."""
    if res.get("markdown") is None:
        raw_chunks = res.get("extracted_chunks") or []
        raw_text = "\n".join(c.raw_text for c in raw_chunks if c.raw_text)
        res["markdown"] = _token_optimizer().optimize_output(
            pii_outputs=res.get("pii_outputs") or [], raw_original_text=raw_text, chunks=raw_chunks
        )
    return res["markdown"]


def _paper_chart(fig, title: str):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title={"text": title, "font": {"family": "IBM Plex Serif, Georgia, serif", "size": 17,
                                       "color": "#1b1b1b"}},
        font={"family": "IBM Plex Mono, Consolas, monospace", "color": "#4a4740", "size": 12},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        showlegend=False,
    )
    fig.update_xaxes(gridcolor="#d8d1c2", linecolor="#1b1b1b")
    fig.update_yaxes(gridcolor="#d8d1c2", linecolor="#1b1b1b")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── 4. MASTHEAD ────────────────────────────────────────────────────────────
head, action = st.columns([5, 1], vertical_alignment="center")
with head:
    st.markdown(ui.masthead(), unsafe_allow_html=True)
with action:
    if st.session_state.res is not None:
        if st.session_state.screen == "case":
            if st.button("Analyse another file", use_container_width=True):
                st.session_state.screen = "intake"
                st.rerun()
        elif st.button("Back to results", use_container_width=True):
            st.session_state.screen = "case"
            st.rerun()
st.markdown('<div class="masthead-rule"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Intake
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "intake" or st.session_state.res is None:
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(ui.intake_heading(), unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload a document (PDF, DOCX or PPTX)",
            type=list(SUPPORTED_EXTENSIONS),
            label_visibility="collapsed",
        )
        go_btn = st.button(
            "Analyse document", type="primary", use_container_width=True, disabled=uploaded is None
        )
        progress = st.empty()

        if go_btn and uploaded:
            started = time.perf_counter()
            stage_reached = {"i": 0}

            def show_progress(current: int, failed: bool = False):
                progress.markdown(
                    ui.live_trail(PIPELINE_STAGES, current, failed,
                                  time.perf_counter() - started, uploaded.name),
                    unsafe_allow_html=True,
                )

            def on_stage(i: int):
                stage_reached["i"] = i
                show_progress(i)

            show_progress(0)
            try:
                res = _run_real_pipeline(uploaded, on_stage=on_stage)
                show_progress(len(PIPELINE_STAGES))
            except (ValueError, OSError, RuntimeError) as exc:
                show_progress(stage_reached["i"], failed=True)
                st.error(f"Processing failed: {exc}")
                st.stop()

            st.session_state.res = res
            st.session_state.viewer_page = 1
            st.session_state.screen = "case"
            st.rerun()

    with right:
        st.markdown(ui.checks_panel(), unsafe_allow_html=True)

    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Results
# ════════════════════════════════════════════════════════════════════════════
res = st.session_state.res
thr = st.session_state.thr
df = res["findings"].copy()
release, review, coverage_pct = verdict(res, thr)
chunks = res.get("extracted_chunks") or []
outputs = res.get("pii_outputs") or []
findings_by_page = {o.page_number: o.detected_entities for o in outputs}
page_count = max(1, len(chunks))
status = res.get("redaction_status")
ext = res.get("file_type", "pdf")

st.markdown(ui.case_header(res, release), unsafe_allow_html=True)
tab_verdict, tab_document, tab_findings, tab_analytics = st.tabs(
    ["Verdict", "Document", "Findings", "Analytics"]
)

# ── Verdict ────────────────────────────────────────────────────────────────
with tab_verdict:
    left, right = st.columns([2, 3], gap="large")
    with left:
        st.markdown(ui.stamp(status), unsafe_allow_html=True)
        uncertainty = {
            "low": sum(c.low_confidence_word_count for c in chunks),
            "unread": sum(len(c.unread_regions) for c in chunks),
            "skipped": sum(c.images_skipped_count for c in chunks),
        }
        st.markdown(
            ui.coverage_meter(res.get("coverage") or {"coverage_pct": coverage_pct},
                              COVERAGE_REQUIRED_PCT, uncertainty),
            unsafe_allow_html=True,
        )
        if review:
            st.markdown(
                ui.note_list("Low-confidence findings",
                             [(f"{review} finding(s) scored below {thr:.2f}; they were redacted, "
                               "but check them in the Findings tab.")], "warn"),
                unsafe_allow_html=True,
            )
        blocking, warns = attention(res)
        st.markdown(ui.note_list("What needs checking", blocking, "block"), unsafe_allow_html=True)
        st.markdown(ui.note_list("Worth a look", warns, "warn"), unsafe_allow_html=True)
    with right:
        st.markdown("### Evidence trail")
        st.markdown(ui.evidence_trail(build_trail(res)), unsafe_allow_html=True)
        st.caption(
            "Cleared means the same detectors found nothing on a fresh read of the redacted "
            "copy. It cannot prove that a value the detectors never recognise is absent."
        )

    st.markdown("### Release")
    sanitized_data: bytes = res.get("sanitized_bytes") or b""
    d1, d2, d3 = st.columns(3)
    with d1:
        label = f"Sanitized {ext.upper()}"
        if status != "VERIFIED_NO_PII_FOUND":
            label += " (not cleared)"
        st.download_button(
            label,
            data=sanitized_data,
            file_name=res["file"].rsplit(".", 1)[0] + f"_SANITIZED.{ext}",
            mime=OUTPUT_MIME.get(ext, "application/octet-stream"),
            disabled=not sanitized_data,
            use_container_width=True,
        )
    with d2:
        try:
            st.download_button(
                "LLM Markdown",
                data=_llm_markdown(res),
                file_name=res["file"].rsplit(".", 1)[0] + "_OPTIMIZED.md",
                mime="text/markdown",
                use_container_width=True,
            )
        except (ValueError, TypeError) as exc:
            st.error(f"Markdown export failed: {exc}")
    with d3:
        st.download_button(
            "Findings CSV",
            data=df.to_csv(index=False).encode(),
            file_name="pii_findings.csv",
            mime="text/csv",
            use_container_width=True,
        )
    if status != "VERIFIED_NO_PII_FOUND":
        st.caption(
            "This copy has not been cleared yet. Check the points in the evidence trail "
            "before sharing it."
        )

# ── Document ───────────────────────────────────────────────────────────────
with tab_document:
    st.session_state.viewer_page = min(max(1, st.session_state.viewer_page), page_count)
    nav, layers = st.columns([1, 3], vertical_alignment="bottom")
    with nav:
        page_no = st.number_input("Page", min_value=1, max_value=page_count, step=1, key="viewer_page")
    chunk = next((c for c in chunks if c.page_number == page_no), None)
    page_findings = findings_by_page.get(page_no, [])

    viewer, side = st.columns([3, 2], gap="large")
    if ext == "pdf" and res.get("original_bytes") and chunk is not None:
        with layers:
            options = ["Original", "Redacted"] if res.get("sanitized_bytes") else ["Original"]
            c_mode, c_find, c_unc = st.columns([2, 1, 1], vertical_alignment="bottom")
            mode = c_mode.segmented_control("View", options, default="Original", key="viewer_mode") or "Original"
            show_findings = c_find.toggle("Findings", value=True, disabled=mode != "Original")
            show_uncertainty = c_unc.toggle("Uncertainty", value=True, disabled=mode != "Original")
        with viewer:
            try:
                if mode == "Redacted":
                    img, _ = _page_image(res["sanitized_bytes"], page_no, VIEWER_DPI)
                else:
                    img, matrix = _page_image(res["original_bytes"], page_no, VIEWER_DPI)
                    img = page_overlay(img, matrix, chunk, page_findings,
                                       findings_layer=show_findings,
                                       uncertainty_layer=show_uncertainty)
                if mode == "Original":
                    st.markdown(ui.legend(), unsafe_allow_html=True)
                st.image(img, width="stretch")
            except (RuntimeError, ValueError, IndexError) as exc:
                st.error(f"This page could not be rendered: {exc}")
    elif chunk is not None:
        with layers:
            st.caption(
                f"{ext.upper()} files have no fixed page layout, so the page is shown as text "
                "with every finding replaced by its token."
            )
        with viewer:
            scanned = scan_chunks([chunk])[0].raw_text
            st.markdown(ui.annotated_text(scanned, page_findings, len(chunk.raw_text)),
                        unsafe_allow_html=True)

    with side:
        if chunk is not None:
            info = page_uncertainty(chunk)
            st.markdown(ui.ledger([
                (len(page_findings), "Findings"),
                (info["low_confidence_words"], "Low-conf words"),
                (info["unread_regions"], "Unread regions"),
                (info["images_skipped"], "Skipped images"),
            ], columns=2), unsafe_allow_html=True)
            if info["not_inspected"]:
                st.markdown(ui.note_list("Not fully inspected", chunk.extraction_errors, "block"),
                            unsafe_allow_html=True)
        st.markdown("#### On this page")
        st.markdown(ui.finding_rows(page_findings), unsafe_allow_html=True)

# ── Findings ───────────────────────────────────────────────────────────────
with tab_findings:
    if df.empty:
        st.info("No PII was detected in this document.")
    else:
        df["Status"] = df["Confidence"].apply(lambda c: "Redacted" if c >= thr else "Needs review")
        c_type, c_src, c_pg = st.columns(3)
        types = c_type.multiselect("PII type", sorted(df["Type"].unique()))
        sources = c_src.multiselect("Source", sorted(df["Source"].unique()))
        page_range = c_pg.slider("Pages", 1, page_count, (1, page_count)) if page_count > 1 else (1, 1)
        view = df[df["Page"].between(*page_range)]
        if types:
            view = view[view["Type"].isin(types)]
        if sources:
            view = view[view["Source"].isin(sources)]
        view = view.reset_index(drop=True)
        st.session_state["_findings_pages"] = view["Page"].tolist()

        def _open_selected_page():
            selection = st.session_state.get("findings_table") or {}
            rows = (selection.get("selection") or {}).get("rows") or []
            pages = st.session_state.get("_findings_pages") or []
            if rows and rows[0] < len(pages):
                st.session_state.viewer_page = int(pages[rows[0]])

        st.caption(f"{len(view)} of {len(df)} findings. Select a row to open its page in the "
                   "Document tab.")
        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
            height=min(420, 36 * (len(view) + 1) + 3),
            key="findings_table",
            on_select=_open_selected_page,
            selection_mode="single-row",
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=1, format="%.2f"
                )
            },
        )
        selected = (st.session_state.get("findings_table") or {}).get("selection", {}).get("rows")
        if selected:
            st.success(f"Page {st.session_state.viewer_page} is open in the Document tab.")

# ── Analytics ──────────────────────────────────────────────────────────────
with tab_analytics:
    raw_text = "\n".join(c.raw_text for c in chunks if c.raw_text)
    conversion = get_conversion_matrix(raw_text, outputs)
    st.markdown(ui.ledger([
        (res.get("pages", 0), "Pages"),
        (len(df), "Findings"),
        (df["Token"].nunique() if not df.empty else 0, "Distinct values"),
        (f"{df['Confidence'].mean():.2f}" if not df.empty else "n/a", "Avg confidence"),
        (f"{conversion['redaction_density_percentage']}%", "Text redacted"),
    ]), unsafe_allow_html=True)

    if df.empty:
        st.info("No PII was detected in this document, so there is nothing to chart.")
    else:
        chart_df = df.assign(Type=df["Type"].str.replace("_", " ").str.title())
        col1, col2 = st.columns(2, gap="large")
        with col1:
            counts = chart_df["Type"].value_counts().sort_values().reset_index()
            counts.columns = ["Type", "Findings"]
            _paper_chart(
                px.bar(counts, x="Findings", y="Type", orientation="h",
                       color_discrete_sequence=[CHART_COLORS[0]]),
                "Findings by type",
            )
        with col2:
            sources = chart_df["Source"].value_counts().reset_index()
            sources.columns = ["Source", "Findings"]
            _paper_chart(
                px.pie(sources, names="Source", values="Findings", hole=0.55,
                       color_discrete_sequence=CHART_COLORS),
                "Where it was found (page text vs image OCR)",
            )
        col3, col4 = st.columns(2, gap="large")
        with col3:
            conf = chart_df.groupby("Type", as_index=False)["Confidence"].mean()
            _paper_chart(
                px.bar(conf, x="Type", y="Confidence", range_y=[0, 1],
                       color_discrete_sequence=[CHART_COLORS[2]]),
                "Average confidence by type",
            )
        with col4:
            per_page = Counter(df["Page"])
            page_df = pd.DataFrame(
                {"Page": list(range(1, page_count + 1)),
                 "Findings": [per_page.get(p, 0) for p in range(1, page_count + 1)]}
            )
            _paper_chart(
                px.bar(page_df, x="Page", y="Findings", color_discrete_sequence=[CHART_COLORS[1]]),
                "Findings per page",
            )
