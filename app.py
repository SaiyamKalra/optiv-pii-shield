"""
PII Sentinel — Frontend
Run:  streamlit run app.py
Plug in the real backend inside _demo_pipeline() — keep the returned dict shape.
"""

import io
import random
import time
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="PII Sentinel",
    page_icon="🛡️",
    layout="wide",
)

# Simple Calibri Font with Bright, High-Contrast Headings for Dark Theme
st.markdown("""
<style>
    html, body, [class*="css"], .stApp {
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif !important;
        background-color: #0d0d14;
        color: #e2e8f0;
    }
    
    /* Bright, High-Contrast Headings */
    h1, h2, h3, h4, h5, h6, 
    [data-testid="stMarkdownContainer"] h1, 
    [data-testid="stMarkdownContainer"] h2, 
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {
        color: #00f2fe !important;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif !important;
        font-weight: 700 !important;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #00f2fe !important;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        margin-bottom: 4px;
        text-shadow: 0 0 12px rgba(0, 242, 254, 0.25);
    }
    .sub-text {
        color: #cbd5e1 !important;
        font-size: 1.05rem;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        margin-bottom: 20px;
    }
    .kpi-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 10px;
        padding: 16px 14px;
        text-align: center;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
    }
    .kpi-num {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8 !important;
    }
    .kpi-label {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .token {
        background: rgba(56, 189, 248, 0.2);
        color: #7dd3fc;
        border: 1px solid rgba(56, 189, 248, 0.4);
        border-radius: 4px;
        padding: 2px 8px;
        font-family: monospace;
        font-size: 0.9rem;
        font-weight: 600;
    }
    .badge-safe    { color: #4ade80; background: rgba(74, 222, 128, 0.15); border: 1px solid rgba(74, 222, 128, 0.4); padding: 4px 14px; border-radius: 16px; font-size: 0.88rem; font-weight: 600; }
    .badge-unknown { color: #facc15; background: rgba(250, 204, 21, 0.15); border: 1px solid rgba(250, 204, 21, 0.4); padding: 4px 14px; border-radius: 16px; font-size: 0.88rem; font-weight: 600; }
    .badge-unsafe  { color: #f87171; background: rgba(248, 113, 113, 0.15); border: 1px solid rgba(248, 113, 113, 0.4); padding: 4px 14px; border-radius: 16px; font-size: 0.88rem; font-weight: 600; }
    
    .info-box {
        background: rgba(255, 255, 255, 0.04);
        border-left: 4px solid #00f2fe;
        border-radius: 6px;
        padding: 16px 20px;
        font-size: 0.98rem;
        color: #f1f5f9;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
    }
    hr { border: none; border-top: 1px solid rgba(255, 255, 255, 0.1); margin: 18px 0; }
    
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Demo backend (replace this with real pipeline) ──────────────────────────
ENTITY_TYPES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN",
                "CREDIT_CARD", "IP_ADDRESS", "LOCATION", "ORGANIZATION"]
WEIGHTS      = [28, 20, 18, 10, 8, 5, 6, 5]
SOURCES      = ["body", "table", "image/OCR", "narrative"]
DETECTORS    = {
    "PERSON":       "spaCy NER",
    "EMAIL_ADDRESS":"Presidio regex",
    "PHONE_NUMBER": "RobustPhone recognizer",
    "US_SSN":       "StrictSSN recognizer",
    "CREDIT_CARD":  "CorpCreditCard recognizer",
    "IP_ADDRESS":   "Presidio regex",
    "LOCATION":     "spaCy NER",
    "ORGANIZATION": "spaCy NER",
}


def _demo_pipeline(filename: str, pages: int) -> dict:
    """
    PLUG-IN POINT — replace with real backend.
    Return dict shape is documented in README.md.
    """
    rng = random.Random(len(filename) * 31)
    rows, counters = [], {}
    for _ in range(rng.randint(90, 160)):
        t = rng.choices(ENTITY_TYPES, weights=WEIGHTS)[0]
        counters[t] = counters.get(t, 0) + 1
        n = counters[t]
        rows.append({
            "Page":       rng.randint(1, pages),
            "Source":     rng.choices(SOURCES, weights=[3, 4, 2, 3])[0],
            "Type":       t,
            "Detector":   DETECTORS[t],
            "Confidence": round(rng.uniform(0.62, 0.99), 3),
            "Token":      f"[{t}_{n:03d}]",
            "Context":    f"Pattern match for '{t.lower().replace('_', ' ')}'",
        })
    df = pd.DataFrame(rows).sort_values(["Page", "Type"]).reset_index(drop=True)
    df.insert(0, "ID", [f"F-{i+1:05d}" for i in df.index])
    ocr_p = pages if filename.lower().endswith(".pdf") else 0
    return {
        "file":            filename,
        "pages":           pages,
        "ocr_pages":       ocr_p,
        "reliable":        max(pages - 1, 1),
        "findings":        df,
        "sanitized_bytes": b"[sanitized document - connect redaction module]",
        "residual":        0,
        "secs":            round(rng.uniform(14, 42), 1),
    }


# ── Helpers ─────────────────────────────────────────────────────────────────
if "res"    not in st.session_state: st.session_state.res    = None
if "screen" not in st.session_state: st.session_state.screen = "upload"
if "thr"    not in st.session_state: st.session_state.thr    = 0.70


def verdict(res, thr):
    df       = res["findings"]
    review   = int((df["Confidence"] < thr).sum())
    coverage = res["reliable"] / res["pages"] * 100
    if res["residual"] > 0:         return "UNSAFE",  review, coverage
    if review > 0 or coverage < 95: return "UNKNOWN", review, coverage
    return "SAFE", review, coverage


def kpi(col, value, label):
    col.markdown(
        f'<div class="kpi-box"><div class="kpi-num">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )


# ── Navigation Header ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
c1.markdown('<div class="main-title">🛡️ PII Sentinel</div>', unsafe_allow_html=True)
with c2:
    if st.button("⬆️ Upload Screen", use_container_width=True):
        st.session_state.screen = "upload"
with c3:
    if st.button("📋 Findings Table", use_container_width=True,
                 disabled=st.session_state.res is None):
        st.session_state.screen = "findings"
with c4:
    if st.button("📊 Analytics Charts", use_container_width=True,
                 disabled=st.session_state.res is None):
        st.session_state.screen = "analytics"

st.markdown("<hr>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — Upload Screen
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.screen == "upload":

    st.markdown('<div class="sub-text">Upload a document to detect and redact sensitive PII before sharing with AI.</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2], gap="large")

    with left:
        uploaded = st.file_uploader(
            "Upload a document (PDF, PPTX, DOCX)",
            type=["pdf", "pptx", "docx"],
        )

        col_a, col_b = st.columns(2)
        go_btn   = col_a.button("🔍 Analyze Document", use_container_width=True,
                                disabled=uploaded is None)
        demo_btn = col_b.button("📄 Try Demo File",   use_container_width=True)

        if go_btn or demo_btn:
            name  = uploaded.name if (go_btn and uploaded) else "Cadence_Risk_Policy_v6.pdf"
            ext   = name.lower().rsplit(".", 1)[-1]
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
                for step in STEPS:
                    st.write(f"✅ {step}")
                    time.sleep(0.3)
                st.session_state.res = _demo_pipeline(name, pages)
                status.update(label="Analysis complete!", state="complete", expanded=False)

            st.session_state.screen = "findings"
            st.rerun()

    with right:
        st.markdown("""
        <div class="info-box">
        <b style="color:#00f2fe; font-size:1.05rem;">Pipeline Steps</b><br><br>
        1. <b style="color:#38bdf8;">Extract Text:</b> Extracts native text and applies OCR to scanned pages.<br>
        2. <b style="color:#38bdf8;">Detect PII:</b> Scans text using Presidio regex and spaCy NER models.<br>
        3. <b style="color:#38bdf8;">Tokenize:</b> Replaces PII with safe tokens like <span class="token">[PERSON_001]</span>.<br>
        4. <b style="color:#38bdf8;">Verify:</b> Re-scans output to ensure 0% residual leak.<br><br>
        <b style="color:#00f2fe;">Supported File Types:</b> PDF, PPTX, DOCX
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#00f2fe !important;'>Sample Redaction Preview</h4>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box" style="font-size:0.95rem; line-height:1.8">
        <b style="color:#f87171;">Original:</b> Contact Rahul Sharma at rahul@gmail.com<br>
        <b style="color:#4ade80;">Redacted:</b> Contact <span class="token">[PERSON_001]</span> at <span class="token">[EMAIL_001]</span>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — Findings Table & Downloads
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "findings" and st.session_state.res:
    res = st.session_state.res
    thr = st.session_state.thr
    df  = res["findings"].copy()
    state, review, coverage = verdict(res, thr)

    BADGE = {"SAFE": "badge-safe", "UNKNOWN": "badge-unknown", "UNSAFE": "badge-unsafe"}
    ICON  = {"SAFE": "✅", "UNKNOWN": "⚠️", "UNSAFE": "❌"}

    st.markdown(
        f'<span style="font-size:1.4rem; font-weight:bold; color:#00f2fe;">{res["file"]}</span> &nbsp;&nbsp; '
        f'<span class="{BADGE[state]}">{ICON[state]} {state}</span> &nbsp;&nbsp; '
        f'<span style="color:#94a3b8; font-size:0.95rem">Processing time: {res["secs"]}s</span>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    # KPI summary row
    k = st.columns(5)
    kpi(k[0], res["pages"],        "Total Pages")
    kpi(k[1], res["ocr_pages"],    "OCR Pages")
    kpi(k[2], len(df),             "PII Findings")
    kpi(k[3], f"{coverage:.1f}%",  "Coverage")
    kpi(k[4], f"{res['secs']}s",   "Processing Time")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Findings List", "🔒 Sanitized Preview"])

    with tab1:
        thr = st.slider("Review confidence threshold",
                        0.5, 0.99, thr, 0.01)
        st.session_state.thr = thr

        c_type, c_src, c_pg = st.columns(3)
        types = c_type.multiselect("Filter PII type", sorted(df["Type"].unique()))
        srcs  = c_src.multiselect("Filter source",    SOURCES)
        pg    = c_pg.slider("Filter page range", 1, res["pages"], (1, res["pages"]))

        df["Status"] = df["Confidence"].apply(
            lambda c: "Redacted" if c >= thr else "Needs review"
        )
        view = df[df["Page"].between(*pg)]
        if types: view = view[view["Type"].isin(types)]
        if srcs:  view = view[view["Source"].isin(srcs)]

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

        st.markdown("<h3 style='color:#00f2fe !important;'>📥 Download Output Files</h3>", unsafe_allow_html=True)
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
                "⬇️ Download Sanitized Text",
                data=res["sanitized_bytes"],
                file_name=res["file"].rsplit(".", 1)[0] + "_SANITIZED.txt",
                use_container_width=True,
            )
        with dl3:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("pii_findings.csv", view.to_csv(index=False))
                zf.writestr("sanitized.txt",     res["sanitized_bytes"])
            buf.seek(0)
            st.download_button(
                "📦 Download All (ZIP)",
                data=buf.getvalue(),
                file_name="pii_sentinel_output.zip",
                mime="application/zip",
                use_container_width=True,
            )

    with tab2:
        b_col, a_col = st.columns(2, gap="large")
        b_col.markdown("<h4 style='color:#38bdf8 !important;'>Original Text (Before Redaction)</h4>", unsafe_allow_html=True)
        b_col.info(
            "Rahul Sharma, employee EMP-38104, can be reached at "
            "rahul@cadence.com or +91-98765-43210. "
            "His SSN is 123-45-6789. Rahul reports to the Group CRO."
        )
        a_col.markdown("<h4 style='color:#00f2fe !important;'>Sanitized Text (After Redaction)</h4>", unsafe_allow_html=True)
        a_col.markdown(
            '<div class="info-box">'
            '<span class="token">[PERSON_001]</span>, employee '
            '<span class="token">[EMPLOYEE_ID_001]</span>, can be reached at '
            '<span class="token">[EMAIL_001]</span> or <span class="token">[PHONE_001]</span>. '
            'His SSN is <span class="token">[US_SSN_001]</span>. '
            '<span class="token">[PERSON_001]</span> reports to the Group CRO.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Analyze another document"):
        st.session_state.screen = "upload"
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — Analytics (Bright Headings & Dark Plotly Charts)
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.screen == "analytics" and st.session_state.res:
    res = st.session_state.res
    df  = res["findings"]
    thr = st.session_state.thr
    state, review, coverage = verdict(res, thr)

    st.markdown(f'<div class="main-title">📊 Analytics Dashboard — {res["file"]}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-text">Graphical analysis of detected PII entities, sources, and confidence scores.</div>', unsafe_allow_html=True)

    # Top KPI summary
    k = st.columns(4)
    kpi(k[0], df["Type"].nunique(),            "PII Categories")
    kpi(k[1], df["Source"].nunique(),          "Source Types")
    kpi(k[2], f"{df['Confidence'].mean():.2f}", "Avg Confidence")
    kpi(k[3], f"{coverage:.1f}%",              "Inspection Coverage")

    st.markdown("<br>", unsafe_allow_html=True)

    # Helper function for dark themed Plotly layout with bright title
    def style_dark_chart(fig, title_text):
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            title={
                "text": f"<b>{title_text}</b>",
                "font": {"color": "#00f2fe", "size": 16, "family": "Calibri, Segoe UI, sans-serif"},
            },
            font={"color": "#e2e8f0", "family": "Calibri, Segoe UI, sans-serif"},
            margin=dict(l=20, r=20, t=50, b=20),
        )
        return fig

    # ── Section 1: Pie Charts ──
    st.markdown("<h3 style='color:#00f2fe !important;'>🥧 Pie Chart Distribution</h3>", unsafe_allow_html=True)
    pie_col1, pie_col2 = st.columns(2, gap="large")

    with pie_col1:
        type_counts = df["Type"].value_counts().reset_index()
        type_counts.columns = ["PII Type", "Count"]
        fig_pie_type = px.pie(
            type_counts,
            names="PII Type",
            values="Count",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            hole=0.35,
        )
        fig_pie_type = style_dark_chart(fig_pie_type, "PII Findings Distribution by Type")
        st.plotly_chart(fig_pie_type, use_container_width=True)

    with pie_col2:
        source_counts = df["Source"].value_counts().reset_index()
        source_counts.columns = ["Source", "Count"]
        fig_pie_source = px.pie(
            source_counts,
            names="Source",
            values="Count",
            color_discrete_sequence=px.colors.qualitative.Set3,
            hole=0.35,
        )
        fig_pie_source = style_dark_chart(fig_pie_source, "PII Findings Distribution by Source")
        st.plotly_chart(fig_pie_source, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 2: Bar Graphs ──
    st.markdown("<h3 style='color:#00f2fe !important;'>📊 Bar Graph Analysis</h3>", unsafe_allow_html=True)
    bar_col1, bar_col2 = st.columns(2, gap="large")

    with bar_col1:
        page_counts = (
            df.groupby("Page").size()
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
        st.plotly_chart(fig_bar_page, use_container_width=True)

    with bar_col2:
        bands = pd.cut(
            df["Confidence"],
            bins=[0, 0.69, 0.84, 1.0],
            labels=["Low (<0.70)", "Medium (0.70-0.84)", "High (>=0.85)"],
        )
        band_counts = bands.value_counts().reindex(
            ["Low (<0.70)", "Medium (0.70-0.84)", "High (>=0.85)"]
        ).reset_index()
        band_counts.columns = ["Confidence Band", "Count"]
        fig_bar_conf = px.bar(
            band_counts,
            x="Confidence Band",
            y="Count",
            color="Confidence Band",
            color_discrete_map={
                "Low (<0.70)": "#f87171",
                "Medium (0.70-0.84)": "#facc15",
                "High (>=0.85)": "#4ade80",
            },
        )
        fig_bar_conf = style_dark_chart(fig_bar_conf, "Findings by Confidence Score Band")
        fig_bar_conf.update_layout(showlegend=False)
        st.plotly_chart(fig_bar_conf, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Section 3: Detector Bar Graph ──
    det_counts = df["Detector"].value_counts().reset_index()
    det_counts.columns = ["Detector Name", "Detections"]
    fig_bar_det = px.bar(
        det_counts,
        x="Detections",
        y="Detector Name",
        orientation="h",
        color_discrete_sequence=["#a855f7"],
    )
    fig_bar_det = style_dark_chart(fig_bar_det, "Detections by Detector Model / Recognizer")
    st.plotly_chart(fig_bar_det, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Export analytics summary button
    summary = pd.DataFrame({
        "Metric": ["Total Findings", "Unique PII Types", "OCR Pages",
                   "Page Coverage", "Avg Confidence", "Residual PII", "Processing Time (s)"],
        "Value":  [len(df), df["Type"].nunique(), res["ocr_pages"],
                   f"{coverage:.1f}%", f"{df['Confidence'].mean():.3f}",
                   res["residual"], res["secs"]],
    })
    st.download_button(
        "⬇️ Export Analytics Summary (CSV)",
        data=summary.to_csv(index=False).encode(),
        file_name="pii_analytics_summary.csv",
        mime="text/csv",
    )

else:
    st.session_state.screen = "upload"
    st.rerun()
