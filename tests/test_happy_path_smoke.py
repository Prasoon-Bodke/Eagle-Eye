"""Smoke test: Happy path through the full Eagle Eye demo flow.

Covers the complete end-to-end scenario a judge would follow:
1. Load sample image + demo checklist
2. Run inspection — check counts (total, PASS, FAIL, Needs Manual Check)
3. Generate summary DataFrame with expected columns
4. Export CSV and Markdown reports
5. Display annotated image (no crash)
6. Simulate fix on first FAIL item and re-inspect
7. Confirm re-inspected summary shows one fewer failure

All assertions use seeded demo data only.
These tests do NOT claim production accuracy.
"""

import sys
from pathlib import Path

# Ensure project root and src/ are importable
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(BASE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(BASE_DIR / "src"))

import cv2
import numpy as np
import pandas as pd
import pytest

from src.eagle_eye.input_processing.file_loader import load_sample_demo_data
from src.eagle_eye.inspection.check_runner import run_inspection
from src.eagle_eye.inspection.result_models import CheckStatus, InspectionSummary
from src.eagle_eye.reporting.report_generator import (
    generate_summary_dataframe,
    export_report_markdown,
)
from src.eagle_eye.reinspection.demo_recheck import simulate_reinspection_fix
from src.eagle_eye.vision.annotation import (
    draw_bounding_box,
    draw_issue_marker,
    draw_demo_evidence_label,
)
from src.eagle_eye.vision.image_processor import save_annotated_image, save_evidence_crops


# ---------------------------------------------------------------------------
# Step 1 — Load sample data
# ---------------------------------------------------------------------------

def test_smoke_load_sample_data():
    """Step 1: Sample image and demo checklist load without error."""
    img, df = load_sample_demo_data()

    assert isinstance(img, np.ndarray), "Sample image must be a NumPy array"
    assert img.ndim == 3 and img.shape[2] == 3, "Image must be H×W×3 BGR"
    assert img.shape[0] > 0 and img.shape[1] > 0, "Image must have non-zero dimensions"

    assert isinstance(df, pd.DataFrame), "Checklist must be a DataFrame"
    assert len(df) >= 5, "Demo checklist must have at least 5 seeded rows"
    required_cols = {"check_id", "check_name", "expected_value"}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"


# ---------------------------------------------------------------------------
# Step 2 — Run inspection and verify counts
# ---------------------------------------------------------------------------

def test_smoke_run_inspection_counts():
    """Step 2: Inspection returns totals including PASS, FAIL, and Needs Manual Check."""
    img, df = load_sample_demo_data()
    summary = run_inspection(image=img, checklist_df=df)

    assert isinstance(summary, InspectionSummary)
    assert summary.total_checks == len(df), "total_checks must equal checklist row count"
    assert summary.total_checks == (
        summary.passed_count + summary.failed_count + summary.manual_check_count
    ), "Counts must add up to total"
    assert summary.passed_count >= 1, "Demo data must have at least one PASS"
    assert summary.failed_count >= 1, "Demo data must have at least one FAIL (seeded)"
    assert summary.manual_check_count >= 1, "Demo data must have at least one Needs Manual Check"


# ---------------------------------------------------------------------------
# Step 3 — Summary DataFrame columns
# ---------------------------------------------------------------------------

def test_smoke_summary_dataframe_columns():
    """Step 3: Summary DataFrame contains all required display columns."""
    img, df = load_sample_demo_data()
    summary = run_inspection(image=img, checklist_df=df)
    summary_df = generate_summary_dataframe(summary)

    required = {"Check ID", "Check Description", "Expected", "Found", "Status", "Source", "Notes"}
    assert required.issubset(summary_df.columns), (
        f"Summary DataFrame missing columns: {required - set(summary_df.columns)}"
    )
    assert len(summary_df) == summary.total_checks


# ---------------------------------------------------------------------------
# Step 4 — Report export (CSV bytes and Markdown string)
# ---------------------------------------------------------------------------

def test_smoke_export_csv_bytes():
    """Step 4a: CSV export produces non-empty UTF-8 bytes with header row."""
    img, df = load_sample_demo_data()
    summary = run_inspection(image=img, checklist_df=df)
    summary_df = generate_summary_dataframe(summary)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    assert len(csv_bytes) > 50, "CSV export must be non-trivially long"
    header_line = csv_bytes.decode("utf-8").splitlines()[0]
    assert "Check ID" in header_line


def test_smoke_export_markdown():
    """Step 4b: Markdown export contains expected sections and summary counts."""
    img, df = load_sample_demo_data()
    summary = run_inspection(image=img, checklist_df=df)
    md = export_report_markdown(summary)

    assert "EAGLE EYE" in md
    assert "Total Checks" in md
    assert "Passed" in md
    assert "Failed" in md
    assert "Detailed Checklist Results" in md


# ---------------------------------------------------------------------------
# Step 5 — Image annotation (no crash)
# ---------------------------------------------------------------------------

def test_smoke_image_annotation(tmp_path):
    """Step 5: Annotating the cabinet image and saving it completes without error."""
    img, df = load_sample_demo_data()
    summary = run_inspection(image=img, checklist_df=df)

    annotated = img.copy()
    for item in summary.items:
        if item.evidence_bbox is not None:
            color = (0, 200, 0) if item.status == CheckStatus.PASS else (0, 0, 220)
            annotated = draw_bounding_box(annotated, item.evidence_bbox, label=item.check_id, color=color)
        if item.marker_id is not None and item.evidence_bbox is not None:
            mx = max(20, item.evidence_bbox[0] + 16)
            my = max(20, item.evidence_bbox[1] + 16)
            annotated = draw_issue_marker(annotated, point=(mx, my), marker_id=item.marker_id)

    annotated = draw_demo_evidence_label(annotated)

    out_path = tmp_path / "smoke_annotated.png"
    saved = save_annotated_image(annotated, out_path)
    assert Path(saved).exists()
    assert Path(saved).stat().st_size > 0

    # Verify crops for FAIL items
    crops = save_evidence_crops(img, summary.items, tmp_path / "crops")
    fail_items = [i for i in summary.items if i.status == CheckStatus.FAIL and i.evidence_bbox is not None]
    assert len(crops) == len(fail_items), "Each FAIL item with bbox should have a crop"


# ---------------------------------------------------------------------------
# Step 6 — Re-inspection: simulate fix and verify before/after
# ---------------------------------------------------------------------------

def test_smoke_reinspection_fix():
    """Step 6: Simulating a fix on first FAIL item changes it to PASS and reduces failure count."""
    img, df = load_sample_demo_data()
    original_summary = run_inspection(image=img, checklist_df=df)

    fail_items = [i for i in original_summary.items if i.status == CheckStatus.FAIL]
    assert len(fail_items) >= 1, "Need at least one FAIL item to simulate fix"

    target = fail_items[0]
    updated_summary = simulate_reinspection_fix(
        original_summary=original_summary,
        target_check_id=target.check_id,
        fixed_found_value=target.expected_value,
        resolution_note="Technician corrected the issue during demo re-inspection.",
    )

    assert updated_summary.total_checks == original_summary.total_checks
    assert updated_summary.failed_count == original_summary.failed_count - 1
    assert updated_summary.passed_count == original_summary.passed_count + 1

    fixed_item = next(i for i in updated_summary.items if i.check_id == target.check_id)
    assert fixed_item.status == CheckStatus.PASS
    assert "RE-INSPECTED" in fixed_item.notes


# ---------------------------------------------------------------------------
# Step 7 — Known Limitations: demo only handles a small subset
# ---------------------------------------------------------------------------

def test_smoke_demo_only_checks_are_seeded():
    """Step 7: Confirm all demo results carry source_type metadata for honest labeling."""
    img, df = load_sample_demo_data()
    summary = run_inspection(image=img, checklist_df=df)

    for item in summary.items:
        assert hasattr(item, "source_type"), f"{item.check_id} missing source_type"
        assert item.source_type in {"demo_seeded", "automatic"}, (
            f"{item.check_id} has unexpected source_type='{item.source_type}'"
        )
