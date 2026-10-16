"""Unit tests for inspection data models, rule runner, and re-inspection.

Owned by: MEMBER 3 (Inspection Logic)
"""

import pandas as pd
from src.eagle_eye.inspection.result_models import (
    CheckStatus,
    InspectionItemResult,
    InspectionSummary,
)
from src.eagle_eye.inspection.check_runner import (
    evaluate_exact_label_match,
    evaluate_missing_part_check,
    evaluate_order_sequence_check,
    evaluate_manual_glare_case,
    evaluate_indicator_state,
    run_check,
    run_checks,
    run_inspection,
    recheck_inspection,
)


# ===========================================================================
# Existing Model & Summary Tests (Preserved)
# ===========================================================================

def test_inspection_summary_counts():
    """Verify that InspectionSummary aggregates statuses accurately."""
    items = [
        InspectionItemResult("CHK-1", "Breaker Q1", "Present", "Present", CheckStatus.PASS),
        InspectionItemResult("CHK-2", "Terminal 1", "TB-01", "Missing", CheckStatus.FAIL),
        InspectionItemResult("CHK-3", "Earth Wire", "PE Green", None, CheckStatus.MANUAL_CHECK),
    ]
    summary = InspectionSummary.from_items(items)

    assert summary.total_checks == 3
    assert summary.passed_count == 1
    assert summary.failed_count == 1
    assert summary.manual_check_count == 1
    assert summary.is_overall_pass() is False


def test_inspection_summary_all_pass():
    """Verify that is_overall_pass() returns True when all checks pass."""
    items = [
        InspectionItemResult("CHK-1", "Breaker Q1", "Present", "Present", CheckStatus.PASS),
        InspectionItemResult("CHK-2", "Terminal 1", "TB-01", "TB-01", CheckStatus.PASS),
    ]
    summary = InspectionSummary.from_items(items)
    assert summary.is_overall_pass() is True


def test_run_inspection_baseline():
    """Verify that the baseline runner processes a DataFrame and returns an InspectionSummary."""
    df = pd.DataFrame([
        {"check_id": "CHK-01", "check_name": "Breaker", "expected_value": "Installed"},
        {"check_id": "CHK-02", "check_name": "Label", "expected_value": "L1"},
    ])
    summary = run_inspection(image=None, checklist_df=df)

    assert summary.total_checks == 2
    assert summary.manual_check_count == 2
    assert len(summary.items) == 2
    assert summary.items[0].check_id == "CHK-01"


# ===========================================================================
# New Rule Runner & Demo Check Unit Tests
# ===========================================================================

def test_exact_label_match_pass():
    """PASS CASE: Exact label match succeeds when observed text matches expected."""
    status, reason, found, source_type = evaluate_exact_label_match(
        expected_label="Q1 63A",
        detected_text="Q1 63A",
    )
    assert status == CheckStatus.PASS
    assert source_type == "automatic"
    assert "Exact label match verified" in reason
    assert found == "Q1 63A"


def test_exact_label_match_fail():
    """FAIL CASE: Exact label match detects discrepancy when observed text conflicts."""
    status, reason, found, source_type = evaluate_exact_label_match(
        expected_label="24VDC",
        detected_text="120VAC",
    )
    assert status == CheckStatus.FAIL
    assert source_type == "automatic"
    assert "Label mismatch" in reason


def test_uncertain_ocr_needs_manual_check():
    """UNCERTAIN CASE: Unreadable/low confidence OCR yields 'Needs Manual Check' (never FAIL)."""
    # Empty OCR text
    status, reason, found, source_type = evaluate_exact_label_match(
        expected_label="TB-07",
        detected_text="",
    )
    assert status == CheckStatus.MANUAL_CHECK
    assert status != CheckStatus.FAIL
    assert source_type == "automatic"
    assert "manual physical verification required" in reason

    # Low confidence / glare keyword
    status_glare, reason_glare, _, _ = evaluate_exact_label_match(
        expected_label="TB-07",
        detected_text="low confidence / glare",
    )
    assert status_glare == CheckStatus.MANUAL_CHECK
    assert status_glare != CheckStatus.FAIL


def test_detector_miss_never_infers_missing_fail():
    """RULE REQUIREMENT: An automated detector miss must NEVER be inferred as a missing part FAIL.

    It must honestly yield 'Needs Manual Check' rather than a confident failure.
    """
    status, reason, found, source_type = evaluate_missing_part_check(
        expected_part="Safety Relay K1",
        found_description=None,
        is_seeded_demo=False,
        detector_miss=True,
    )
    assert status == CheckStatus.MANUAL_CHECK
    assert status != CheckStatus.FAIL
    assert source_type == "automatic"
    assert "Detector absence alone does not infer a missing component" in reason


def test_seeded_missing_part_fail():
    """FAIL CASE: Seeded demo missing/disconnected component yields FAIL with source_type demo_seeded."""
    status, reason, found, source_type = evaluate_missing_part_check(
        expected_part="PE Ground Bonding",
        found_description="Disconnected / Open Lug",
        is_seeded_demo=True,
        detector_miss=False,
    )
    assert status == CheckStatus.FAIL
    assert source_type == "demo_seeded"
    assert "Seeded demo defect" in reason


def test_order_sequence_fail_and_pass():
    """ORDER CHECK: Verifies correct sequence detection and order mismatch failure."""
    # PASS: Correct order
    status_pass, reason_pass, _, src_pass = evaluate_order_sequence_check(
        expected_order="TB-01, TB-02, TB-03",
        found_order="TB-01, TB-02, TB-03",
    )
    assert status_pass == CheckStatus.PASS
    assert src_pass == "automatic"

    # FAIL: Order mismatch (inverted)
    status_fail, reason_fail, _, src_fail = evaluate_order_sequence_check(
        expected_order="TB-01, TB-02, TB-03",
        found_order="TB-01, TB-03, TB-02",
    )
    assert status_fail == CheckStatus.FAIL
    assert src_fail == "automatic"
    assert "Order mismatch" in reason_fail


def test_seeded_manual_check_glare_case():
    """UNCERTAIN CASE: Seeded manual check case models visual limitations honestly."""
    status, reason, found, source_type = evaluate_manual_glare_case(
        expected_item="Terminal Block Numbers",
        found_description="Glare obscures TB-07..09",
        notes="Surface glare prevents OCR confidence",
    )
    assert status == CheckStatus.MANUAL_CHECK
    assert source_type == "demo_seeded"
    assert "Sample Demo: Flagged for Human Sign-off" in reason


def test_run_check_schema_fields():
    """SCHEMA CONTRACT: Verify run_check returns check_id, check_name, expected, found, status, reason, evidence_box, source_type."""
    row = {
        "check_id": "CHK-T1",
        "check_name": "Voltage Label",
        "expected_value": "24VDC",
        "found_value": "24VDC",
        "check_type": "exact_label_match",
        "evidence_box": "100,120,200,220",
        "notes": "Verify 24V supply",
    }
    item = run_check(row)

    # Check attribute and property aliases
    assert item.check_id == "CHK-T1"
    assert item.check_name == "Voltage Label"
    assert item.expected == "24VDC"
    assert item.expected_value == "24VDC"
    assert item.found == "24VDC"
    assert item.found_value == "24VDC"
    assert item.status == CheckStatus.PASS
    assert isinstance(item.reason, str)
    assert len(item.reason) > 0
    assert item.reason == item.notes
    assert item.evidence_box == (100, 120, 200, 220)
    assert item.evidence_bbox == (100, 120, 200, 220)
    assert item.source_type == "automatic"

    # Check dictionary access and serialization
    d = item.to_dict()
    assert d["check_id"] == "CHK-T1"
    assert d["expected"] == "24VDC"
    assert d["found"] == "24VDC"
    assert d["status"] == "PASS"
    assert d["reason"] == item.reason
    assert d["evidence_box"] == (100, 120, 200, 220)
    assert d["source_type"] == "automatic"
    assert item["check_id"] == "CHK-T1"
    assert item["expected"] == "24VDC"


def test_recheck_inspection_simulated_fix():
    """RE-INSPECTION: Demonstrates one issue transitioning from FAIL to PASS after simulated fix."""
    items = [
        InspectionItemResult(
            check_id="CHK-001",
            check_name="Breaker",
            expected_value="Installed",
            found_value="Installed",
            status=CheckStatus.PASS,
            notes="Verified",
            source_type="automatic",
        ),
        InspectionItemResult(
            check_id="CHK-002",
            check_name="PE Ground Lug",
            expected_value="Connected",
            found_value="Disconnected / Open Lug",
            status=CheckStatus.FAIL,
            notes="Ground wire disconnected",
            source_type="demo_seeded",
        ),
    ]
    original_summary = InspectionSummary.from_items(items)

    assert original_summary.total_checks == 2
    assert original_summary.passed_count == 1
    assert original_summary.failed_count == 1
    assert original_summary.is_overall_pass() is False

    # Perform re-check with simulated fix
    updated_summary = recheck_inspection(
        original_summary=original_summary,
        target_check_id="CHK-002",
        fixed_found_value="Connected (Torqued to specification)",
        resolution_reason="Technician secured ground lug with rated torque.",
    )

    assert updated_summary.total_checks == 2
    assert updated_summary.failed_count == 0
    assert updated_summary.passed_count == 2
    assert updated_summary.is_overall_pass() is True

    resolved_item = [i for i in updated_summary.items if i.check_id == "CHK-002"][0]
    assert resolved_item.status == CheckStatus.PASS
    assert resolved_item.found == "Connected (Torqued to specification)"
    assert "RE-INSPECTED: Technician secured ground lug with rated torque." in resolved_item.reason
