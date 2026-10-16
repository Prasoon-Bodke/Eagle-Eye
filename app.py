"""EAGLE EYE — AI Vision for Cabinet Quality Control.

Interactive Hackathon Demonstration Application.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Ensure project root and src/ are in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

from config.settings import (
    APP_TITLE,
    APP_TAGLINE,
    APP_VERSION,
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_CHECKLIST_EXTENSIONS,
    DEFAULT_CHECKLIST_PATH,
    DEFAULT_DEMO_CHECKLIST_PATH,
    DEFAULT_SAMPLE_IMAGE_PATH,
)
from src.eagle_eye.input_processing.file_loader import (
    load_sample_demo_data,
    validate_image_file,
    validate_checklist_file,
    get_sample_image_path,
    get_sample_checklist_path,
)
from src.eagle_eye.vision.annotation import (
    draw_bounding_box,
    draw_issue_marker,
    draw_demo_evidence_label,
)
from src.eagle_eye.vision.image_processor import (
    save_annotated_image,
    save_evidence_crops,
)
from src.eagle_eye.inspection.result_models import (
    CheckStatus,
    InspectionSummary,
)
from src.eagle_eye.inspection.check_runner import (
    run_inspection,
)
from src.eagle_eye.reporting.report_generator import (
    generate_summary_dataframe,
    export_report_markdown,
)
from src.eagle_eye.reinspection.demo_recheck import (
    simulate_reinspection_fix,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EAGLE EYE — Cabinet QC Prototype",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #0d6efd;
        margin-bottom: 12px;
    }
    .status-pass  { color: #198754; font-weight: bold; }
    .status-fail  { color: #dc3545; font-weight: bold; }
    .status-manual{ color: #fd7e14; font-weight: bold; }
    .before-after-card {
        background: #f1f3f5;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        border-left: 4px solid #6c757d;
    }
    .marker-highlight {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 4px 10px;
        border-radius: 4px;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# Header & Branding
# ──────────────────────────────────────────────────────────────────────────────
st.title("🦅 EAGLE EYE — Cabinet QC Prototype")
st.caption("*Prototype / not production QC — Hackathon proof-of-concept only*")

# Prototype Honesty Banner
st.info(
    "🔬 **HACKATHON PROOF-OF-CONCEPT DEMONSTRATION**\n\n"
    "This prototype demonstrates an AI-assisted quality control workflow for industrial control cabinets. "
    "Automated findings are seeded from prepared verification samples and rule-based checks. "
    "**Eagle Eye never claims 100% accuracy**; ambiguous visual features are marked as "
    "**Needs Manual Check** for human inspector sign-off.",
    icon="ℹ️",
)

# ──────────────────────────────────────────────────────────────────────────────
# Session State Initialisation
# ──────────────────────────────────────────────────────────────────────────────
_defaults = {
    "cabinet_image": None,
    "image_name": None,
    "checklist_df": None,
    "checklist_name": None,
    "inspection_summary": None,
    "pre_fix_summary": None,   # stored for before/after comparison
    "reinspected": False,
    "fixed_check_id": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚡ Demo Quick Start")
    st.markdown(
        "Click below to load pre-configured sample data immediately:"
    )

    if st.button("🚀 Load Sample Demo Data", use_container_width=True, type="primary"):
        try:
            demo_img, demo_df = load_sample_demo_data()
            st.session_state["cabinet_image"] = demo_img
            st.session_state["image_name"] = "sample_cabinet.png (Pre-loaded Synthetic Demo)"
            st.session_state["checklist_df"] = demo_df
            st.session_state["checklist_name"] = "sample_demo_checklist.csv (Pre-loaded Seeded Demo)"
            st.session_state["inspection_summary"] = None
            st.session_state["pre_fix_summary"] = None
            st.session_state["reinspected"] = False
            st.session_state["fixed_check_id"] = None
            st.success("✅ Sample demo image and checklist loaded!")
        except Exception as e:
            st.error(f"Failed to load sample demo: {e}")

    st.markdown("---")
    st.subheader("📁 Sample Assets")
    st.caption(
        "- **Cabinet Image**: `sample_data/sample_cabinet.png`\n"
        "- **Checklist CSV**: `sample_data/sample_demo_checklist.csv`\n"
        "- **Checklist Excel**: `sample_data/sample_demo_checklist.xlsx`\n"
        "- **Template**: `sample_data/checklist_template.csv`"
    )

    st.markdown("---")
    st.subheader("⚙️ System Status")
    st.caption(
        f"- **Version**: `{APP_VERSION}`\n"
        f"- **Image Formats**: `{', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}`\n"
        f"- **Checklist Formats**: `{', '.join(sorted(SUPPORTED_CHECKLIST_EXTENSIONS))}`\n"
        f"- **Privacy**: Sanitized synthetic data only"
    )

    if st.button("🔄 Reset Demo State", use_container_width=True):
        for k, v in _defaults.items():
            st.session_state[k] = v
        st.rerun()

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — Input Intake
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 1️⃣ Input Intake: Photo & Checklist")

col_upload_left, col_upload_right = st.columns(2)

with col_upload_left:
    st.markdown("### 📷 Cabinet Photograph")
    uploaded_image_file = st.file_uploader(
        "Upload Cabinet Photo (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
        help="Upload a frontal high-resolution photograph of the manufactured cabinet interior.",
    )

    if uploaded_image_file is not None:
        valid_img, err_img, decoded_img = validate_image_file(
            uploaded_image_file, filename=uploaded_image_file.name
        )
        if valid_img and decoded_img is not None:
            st.session_state["cabinet_image"] = decoded_img
            st.session_state["image_name"] = uploaded_image_file.name
            st.session_state["inspection_summary"] = None
            st.session_state["pre_fix_summary"] = None
            st.session_state["reinspected"] = False
            st.session_state["fixed_check_id"] = None
            st.success(f"Loaded: `{uploaded_image_file.name}` ({decoded_img.shape[1]}x{decoded_img.shape[0]} px)")
        else:
            st.error(f"⚠️ Image Validation Error: {err_img}")

    if st.session_state["cabinet_image"] is not None:
        st.caption(f"Active image: **{st.session_state['image_name']}**")
        thumb_rgb = cv2.cvtColor(st.session_state["cabinet_image"], cv2.COLOR_BGR2RGB)
        st.image(thumb_rgb, caption="Cabinet Photo Preview", use_container_width=True)
    else:
        st.info("No cabinet photograph loaded yet. Upload above or click 'Load Sample Demo Data' in the sidebar.")

with col_upload_right:
    st.markdown("### 📋 Expected Engineering Checklist")
    uploaded_checklist_file = st.file_uploader(
        "Upload Checklist (CSV or Excel .xlsx)",
        type=["csv", "xlsx", "xls"],
        key="checklist_uploader",
        help="Upload component schedule with required columns: check_id, check_name, expected_value.",
    )

    if uploaded_checklist_file is not None:
        valid_chk, err_chk, parsed_df = validate_checklist_file(
            uploaded_checklist_file, filename=uploaded_checklist_file.name
        )
        if valid_chk and parsed_df is not None:
            st.session_state["checklist_df"] = parsed_df
            st.session_state["checklist_name"] = uploaded_checklist_file.name
            st.session_state["inspection_summary"] = None
            st.session_state["pre_fix_summary"] = None
            st.session_state["reinspected"] = False
            st.session_state["fixed_check_id"] = None
            st.success(f"Loaded: `{uploaded_checklist_file.name}` ({len(parsed_df)} checks)")
        else:
            st.error(f"⚠️ Checklist Validation Error: {err_chk}")

    if st.session_state["checklist_df"] is not None:
        st.caption(f"Active checklist: **{st.session_state['checklist_name']}**")
        preview_cols = [c for c in ["check_id", "check_name", "expected_value", "demo_status"] if c in st.session_state["checklist_df"].columns]
        st.dataframe(st.session_state["checklist_df"][preview_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No checklist loaded yet. Upload above or click 'Load Sample Demo Data' in the sidebar.")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — Run Inspection
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 2️⃣ Quality Inspection Execution")

ready_to_inspect = (
    st.session_state["cabinet_image"] is not None
    and st.session_state["checklist_df"] is not None
)

if not ready_to_inspect:
    st.warning("⚠️ Please provide both a Cabinet Photo and an Engineering Checklist before running inspection.")
else:
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("🔍 Run Demo Inspection", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Analysing cabinet photo and evaluating engineering checklist…"):
            summary = run_inspection(
                image=st.session_state["cabinet_image"],
                checklist_df=st.session_state["checklist_df"],
            )
            st.session_state["inspection_summary"] = summary
            st.session_state["pre_fix_summary"] = None
            st.session_state["reinspected"] = False
            st.session_state["fixed_check_id"] = None

# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — Results
# ──────────────────────────────────────────────────────────────────────────────
if st.session_state["inspection_summary"] is not None:
    summary: InspectionSummary = st.session_state["inspection_summary"]

    st.markdown("---")
    st.markdown("## 3️⃣ Inspection Results & Visual Discrepancy Evidence")

    if st.session_state["reinspected"]:
        fixed_id = st.session_state.get("fixed_check_id", "")
        st.success(
            f"🎉 **RE-INSPECTION UPDATED** — Simulated fix on `{fixed_id}` verified. "
            "See before/after comparison below."
        )

    # ── KPI Cards ────────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Checks", summary.total_checks)
    kpi2.metric("✅ Passed", summary.passed_count)
    kpi3.metric("❌ Failed", summary.failed_count)
    kpi4.metric("🔍 Manual Check", summary.manual_check_count)
    overall_label = "PASS ✅" if summary.is_overall_pass() else "ACTION REQUIRED ⚠️"
    kpi5.metric("Overall Outcome", overall_label)

    st.caption(
        "ℹ️ These counts reflect the **current inspection run** only — "
        "they are not accuracy metrics for the underlying AI system."
    )

    # ── Detailed Results Table ────────────────────────────────────────────────
    st.markdown("### 📊 Expected vs. Found Findings")

    summary_df = generate_summary_dataframe(summary)

    # Rename Source column to friendlier label
    summary_df["Source"] = summary_df["Source"].map(
        lambda s: "🤖 Auto" if s == "automatic" else "🌱 Demo-Seeded"
    )

    def _color_status(val: str) -> str:
        if val == "PASS":
            return "background-color: #d1e7dd; color: #0f5132; font-weight: bold;"
        elif val == "FAIL":
            return "background-color: #f8d7da; color: #842029; font-weight: bold;"
        return "background-color: #fff3cd; color: #664d03; font-weight: bold;"

    styled_df = summary_df.style.map(_color_status, subset=["Status"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ── Annotated Image + Marker Legend ─────────────────────────────────────
    st.markdown("### 🔎 Physical Visual Evidence & Issue Markers")
    st.caption(
        "Numbered markers on the photo correspond to rows in the table above. "
        "🔴 = FAIL  🟠 = Needs Manual Check  🟢 = PASS"
    )

    col_img_anno, col_legend = st.columns([3, 2])

    with col_img_anno:
        annotated_img = st.session_state["cabinet_image"].copy()
        markers_info = []   # (marker_id, check_id, check_name, status_val, notes)

        for item in summary.items:
            if item.evidence_bbox is not None:
                if item.status == CheckStatus.PASS:
                    box_color = (0, 200, 0)
                elif item.status == CheckStatus.FAIL:
                    box_color = (0, 0, 220)
                else:
                    box_color = (0, 165, 255)

                label_text = f"{item.check_id}: {item.status.value}"
                annotated_img = draw_bounding_box(
                    annotated_img,
                    item.evidence_bbox,
                    label=label_text,
                    color=box_color,
                    thickness=3,
                )

                if item.marker_id is not None:
                    mx = max(20, item.evidence_bbox[0] + 16)
                    my = max(20, item.evidence_bbox[1] + 16)
                    if item.status == CheckStatus.FAIL:
                        m_color = (0, 0, 220)
                    elif item.status == CheckStatus.MANUAL_CHECK:
                        m_color = (0, 165, 255)
                    else:
                        m_color = (0, 180, 0)
                    annotated_img = draw_issue_marker(
                        annotated_img,
                        point=(mx, my),
                        marker_id=item.marker_id,
                        color=m_color,
                        radius=14,
                    )
                    markers_info.append(
                        (item.marker_id, item.check_id, item.check_name, item.status.value, item.notes)
                    )

        annotated_img = draw_demo_evidence_label(annotated_img)

        # Save to disk
        save_annotated_image(
            annotated_img,
            BASE_DIR / "outputs" / "annotated_cabinet.png",
        )
        crop_paths = save_evidence_crops(
            st.session_state["cabinet_image"],
            summary.items,
            BASE_DIR / "outputs" / "evidence_crops",
        )

        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        st.image(
            annotated_rgb,
            caption="Cabinet Photograph — Annotated with Numbered Quality Markers (SAMPLE / DEMO EVIDENCE)",
            use_container_width=True,
        )

    with col_legend:
        st.markdown("#### 🏷️ Marker Legend")
        st.caption("Each number matches a row in the findings table.")
        if markers_info:
            for mid, cid, cname, stat, notes in markers_info:
                if stat == "FAIL":
                    badge = "🔴 FAIL"
                elif stat == "Needs Manual Check":
                    badge = "🟠 NEEDS MANUAL CHECK"
                else:
                    badge = "🟢 PASS"
                st.markdown(
                    f"<div class='marker-highlight'>"
                    f"<b>Marker #{mid}</b> [{badge}]<br>"
                    f"<code>{cid}</code> — {cname}<br>"
                    f"<small><i>{notes[:120]}{'…' if len(notes) > 120 else ''}</i></small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No localised visual bounding boxes for current items.")

        # Evidence crops for FAIL items
        if crop_paths:
            st.markdown("#### 📸 FAIL Evidence Crops")
            for cid, cpath in crop_paths.items():
                crop_img = cv2.imread(cpath)
                if crop_img is not None:
                    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    st.image(crop_rgb, caption=f"Evidence crop: {cid} (demo/sample)", width=240)

    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 4 — Export + Re-inspection (side by side)
    # ──────────────────────────────────────────────────────────────────────────
    col_export, col_recheck = st.columns(2)

    with col_export:
        st.markdown("## 4️⃣ Export Inspection Report")
        st.markdown("Download auditable quality assurance records:")

        csv_data = summary_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download CSV Inspection Report",
            data=csv_data,
            file_name="eagle_eye_inspection_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        md_report = export_report_markdown(summary)
        st.download_button(
            label="📝 Download Markdown Audit Summary",
            data=md_report.encode("utf-8"),
            file_name="eagle_eye_inspection_summary.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_recheck:
        st.markdown("## 5️⃣ Closed-Loop Re-Inspection")
        st.markdown(
            "Simulate a technician correcting a flagged defect and re-verifying the cabinet. "
            "The before/after comparison will appear below."
        )

        failing_items = [item for item in summary.items if item.status == CheckStatus.FAIL]

        if failing_items:
            target_item = failing_items[0]
            st.caption(
                f"Flagged item for fix: `{target_item.check_id}` — **{target_item.check_name}**\n\n"
                f"*(Current finding: {target_item.found_value})*"
            )
            fix_btn_label = f"🔧 Simulate Fix: Resolve {target_item.check_id} & Re-Inspect"
            if st.button(fix_btn_label, type="secondary", use_container_width=True):
                pre_fix = st.session_state["inspection_summary"]
                updated_summary = simulate_reinspection_fix(
                    original_summary=pre_fix,
                    target_check_id=target_item.check_id,
                    fixed_found_value=target_item.expected_value,
                    resolution_note=(
                        "Field technician secured PE bonding lug to ground busbar with rated torque."
                    ),
                )
                st.session_state["pre_fix_summary"] = pre_fix
                st.session_state["inspection_summary"] = updated_summary
                st.session_state["reinspected"] = True
                st.session_state["fixed_check_id"] = target_item.check_id
                st.rerun()
        else:
            st.success("✅ All failures resolved — no active failures pending re-inspection.")

    # ── Before / After Comparison ─────────────────────────────────────────────
    if st.session_state["reinspected"] and st.session_state["pre_fix_summary"] is not None:
        pre: InspectionSummary = st.session_state["pre_fix_summary"]
        post: InspectionSummary = st.session_state["inspection_summary"]
        fixed_id = st.session_state.get("fixed_check_id", "")

        st.markdown("### 🔄 Before / After Re-Inspection Comparison")
        ba_left, ba_right = st.columns(2)

        with ba_left:
            st.markdown(
                f"<div class='before-after-card'>"
                f"<b>BEFORE fix</b><br>"
                f"Total: {pre.total_checks} | "
                f"✅ {pre.passed_count} | "
                f"❌ {pre.failed_count} | "
                f"🔍 {pre.manual_check_count}"
                f"</div>",
                unsafe_allow_html=True,
            )
            pre_item = next((i for i in pre.items if i.check_id == fixed_id), None)
            if pre_item:
                st.markdown(
                    f"**`{fixed_id}`** status: `{pre_item.status.value}` → "
                    f"found: *{pre_item.found_value}*"
                )

        with ba_right:
            st.markdown(
                f"<div class='before-after-card' style='border-left-color:#198754;'>"
                f"<b>AFTER fix</b><br>"
                f"Total: {post.total_checks} | "
                f"✅ {post.passed_count} | "
                f"❌ {post.failed_count} | "
                f"🔍 {post.manual_check_count}"
                f"</div>",
                unsafe_allow_html=True,
            )
            post_item = next((i for i in post.items if i.check_id == fixed_id), None)
            if post_item:
                st.markdown(
                    f"**`{fixed_id}`** status: `{post_item.status.value}` → "
                    f"found: *{post_item.found_value}*"
                )

    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────────────
    # Section 6 — Known Limitations
    # ──────────────────────────────────────────────────────────────────────────
    with st.expander("⚠️ Known Limitations — Please Read Before Judging", expanded=False):
        st.markdown(
            """
### Known Limitations of This Prototype

This is a **hackathon prototype** built in a short timeframe for demonstration purposes only.

| Limitation | Detail |
|---|---|
| **Demo-seeded results** | Most check results are pre-seeded from `sample_demo_checklist.csv`. They illustrate the workflow but are not live AI detections. |
| **Partial component coverage** | The demo covers only 5 checklist items. A production system would need hundreds of checks. |
| **No real wiring verification** | Wire routing, conductor colour coding, and terminal torque values are **not verified** by this prototype. |
| **No drawing-to-photo alignment** | The prototype does not align electrical schematics to the cabinet photo. |
| **No trained object-detection model** | Object detection is not integrated; bounding boxes come from checklist metadata. |
| **OCR requires Tesseract** | Label reading via OCR falls back to "Needs Manual Check" when Tesseract is not installed — which is correct and honest behaviour. |
| **No formal accuracy benchmarking** | We do **not** claim detection accuracy metrics. Counts shown are per-run totals, not model performance. |
| **Synthetic sample image** | The cabinet photo is a programmatically generated synthetic image, not a real factory photograph. |
| **Single session, no persistence** | All state is held in Streamlit session memory; there is no database or audit log. |

> **Summary**: This prototype handles a few clearly-labelled demo checks and does not yet verify all wiring, all cabinet components, or terminal labelling at production scale. Results labelled **"Demo-Seeded"** are illustrative; results labelled **"Auto"** reflect rule-based logic applied to the checklist values.
            """
        )
