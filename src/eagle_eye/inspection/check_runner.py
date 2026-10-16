"""Core inspection execution runner and rule-based checks.

Owned by: MEMBER 3 (Inspection Logic)
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from .result_models import CheckStatus, InspectionItemResult, InspectionSummary


# ===========================================================================
# 3–5 Clear Rule-Based Check Implementations
# ===========================================================================

def evaluate_exact_label_match(
    expected_label: str,
    detected_text: Optional[str],
) -> Tuple[CheckStatus, str, Optional[str], str]:
    """Check 1: Exact label match verification.

    Compares expected engineering label against observed/OCR text.
    - Matches normalized text -> PASS (source_type = 'automatic')
    - Explicit contradiction -> FAIL (source_type = 'automatic')
    - Missing or unreadable text -> Needs Manual Check (source_type = 'automatic')

    Rule: Never turn low OCR confidence or unreadable text alone into FAIL.
    """
    if detected_text is None or str(detected_text).strip() == "" or str(detected_text).strip().lower() in {"pending", "pending verification", "n/a", "none"}:
        return (
            CheckStatus.MANUAL_CHECK,
            f"Label reading inconclusive or unreadable for expected '{expected_label}'; manual physical verification required.",
            detected_text if detected_text else "Unreadable / Inconclusive",
            "automatic",
        )

    norm_detected = " ".join(str(detected_text).strip().lower().split())
    norm_expected = " ".join(str(expected_label).strip().lower().split())

    # Check for keywords indicating OCR / visibility impediment
    if any(term in norm_detected for term in ["unreadable", "obscured", "low confidence", "glare", "blurry"]):
        return (
            CheckStatus.MANUAL_CHECK,
            f"Visual glare or low OCR confidence on label '{expected_label}'; requires manual verification.",
            str(detected_text),
            "automatic",
        )

    if norm_detected == norm_expected or norm_expected in norm_detected:
        return (
            CheckStatus.PASS,
            f"Exact label match verified: expected '{expected_label}', found '{detected_text}'.",
            str(detected_text),
            "automatic",
        )

    return (
        CheckStatus.FAIL,
        f"Label mismatch: expected '{expected_label}', but detected '{detected_text}'.",
        str(detected_text),
        "automatic",
    )


def evaluate_missing_part_check(
    expected_part: str,
    found_description: Optional[str],
    is_seeded_demo: bool = False,
    detector_miss: bool = False,
) -> Tuple[CheckStatus, str, Optional[str], str]:
    """Check 2: Component presence / Seeded missing part check.

    Rule: Never infer a missing component solely because an object detector did not find it.
    A detector miss alone yields "Needs Manual Check".
    FAIL is reserved for positive evidence of absence or explicitly confirmed demo flaws.
    """
    if detector_miss:
        return (
            CheckStatus.MANUAL_CHECK,
            "Component not confidently identified by automated detector; requires manual physical verification. "
            "(Detector absence alone does not infer a missing component)",
            found_description or "Detector Miss (Uncertain)",
            "automatic",
        )

    found_str = str(found_description or "").strip()
    norm_found = found_str.lower()

    if is_seeded_demo:
        if any(term in norm_found for term in ["missing", "disconnected", "open lug", "absent", "omitted"]):
            return (
                CheckStatus.FAIL,
                f"Seeded demo defect: {found_str}",
                found_str,
                "demo_seeded",
            )
        elif any(term in norm_found for term in ["present", "connected", "installed", "ok"]):
            return (
                CheckStatus.PASS,
                f"Component presence verified: {found_str}",
                found_str,
                "demo_seeded",
            )
        else:
            return (
                CheckStatus.MANUAL_CHECK,
                f"Evidence inconclusive for '{expected_part}'; requires manual check [Sample Demo].",
                found_str or "Pending Verification",
                "demo_seeded",
            )

    # Automatic evaluation
    if not found_str or norm_found in {"pending", "pending verification", "n/a", "none"}:
        return (
            CheckStatus.MANUAL_CHECK,
            "Visual presence inconclusive; requires manual inspection (never inferred missing from absence of detection).",
            found_str or "Pending Verification",
            "automatic",
        )

    if any(term in norm_found for term in ["present", "installed", "ok", "mounted", "connected"]):
        return (
            CheckStatus.PASS,
            f"Component verified present: {found_str}",
            found_str,
            "automatic",
        )

    if any(term in norm_found for term in ["empty slot", "confirmed missing", "disconnected"]):
        return (
            CheckStatus.FAIL,
            f"Component discrepancy confirmed: {found_str}",
            found_str,
            "automatic",
        )

    return (
        CheckStatus.MANUAL_CHECK,
        f"Ambiguous presence indication for '{expected_part}'; requires manual check.",
        found_str,
        "automatic",
    )


def evaluate_order_sequence_check(
    expected_order: str,
    found_order: Optional[str],
) -> Tuple[CheckStatus, str, Optional[str], str]:
    """Check 3: Component / Terminal order sequence check.

    Verifies sequential terminal numbering or wiring order.
    """
    found_str = str(found_order or "").strip()
    norm_found = found_str.lower()

    if not found_str or any(term in norm_found for term in ["pending", "unreadable", "obscured", "unclear"]):
        return (
            CheckStatus.MANUAL_CHECK,
            "Sequence markings partially obscured or unreadable; requires manual inspection to verify order.",
            found_str or "Obscured / Inconclusive",
            "automatic",
        )

    # Detect explicit order mismatch indications
    if any(term in norm_found for term in ["mismatch", "inverted", "out of order", "wrong order", "reversed"]):
        return (
            CheckStatus.FAIL,
            f"Order mismatch detected: expected sequential [{expected_order}], but found [{found_str}].",
            found_str,
            "automatic",
        )

    # Check comma-separated lists
    if "," in expected_order and "," in found_str:
        exp_tokens = [t.strip().lower() for t in expected_order.split(",") if t.strip()]
        fnd_tokens = [t.strip().lower() for t in found_str.split(",") if t.strip()]
        if exp_tokens == fnd_tokens:
            return (
                CheckStatus.PASS,
                f"Sequential order verified: [{found_str}].",
                found_str,
                "automatic",
            )
        else:
            return (
                CheckStatus.FAIL,
                f"Order mismatch: expected [{expected_order}], but found [{found_str}].",
                found_str,
                "automatic",
            )

    # Normalized text equality
    if norm_found == expected_order.strip().lower() or "sequential" in norm_found:
        return (
            CheckStatus.PASS,
            f"Sequential order verified: {found_str}.",
            found_str,
            "automatic",
        )

    return (
        CheckStatus.MANUAL_CHECK,
        f"Unable to confirm terminal order '{expected_order}'; requires manual check.",
        found_str,
        "automatic",
    )


def evaluate_manual_glare_case(
    expected_item: str,
    found_description: Optional[str] = None,
    notes: str = "Surface glare / occlusion prevents confident automated reading.",
) -> Tuple[CheckStatus, str, Optional[str], str]:
    """Check 4: Seeded uncertainty / Needs Manual Check demonstration.

    Represents physical lighting, reflections, or occlusion.
    """
    found = found_description or "Visual Glare / Obscured"
    reason = f"{notes} [Sample Demo: Flagged for Human Sign-off]".strip()
    return (CheckStatus.MANUAL_CHECK, reason, found, "demo_seeded")


def evaluate_indicator_state(
    expected_state: str,
    found_state: Optional[str],
) -> Tuple[CheckStatus, str, Optional[str], str]:
    """Check 5: Operational status indicator check (e.g. 24VDC LED Green, PLC RUN)."""
    found_str = str(found_state or "").strip()
    norm_found = found_str.lower()
    norm_expected = expected_state.strip().lower()

    if not found_str or norm_found in {"pending", "unclear", "unreadable", "n/a"}:
        return (
            CheckStatus.MANUAL_CHECK,
            f"Indicator status unreadable for '{expected_state}'; manual check required.",
            found_str or "Pending Verification",
            "automatic",
        )

    if norm_found == norm_expected or norm_expected in norm_found:
        return (
            CheckStatus.PASS,
            f"Indicator verified: expected '{expected_state}', observed '{found_str}'.",
            found_str,
            "automatic",
        )

    if any(term in norm_found for term in ["off", "red", "fault", "error", "alarm", "disconnected"]):
        return (
            CheckStatus.FAIL,
            f"Indicator discrepancy: expected '{expected_state}', but observed '{found_str}'.",
            found_str,
            "automatic",
        )

    return (
        CheckStatus.MANUAL_CHECK,
        f"Ambiguous indicator reading for '{expected_state}'; requires manual inspection.",
        found_str,
        "automatic",
    )


# ===========================================================================
# Check Runner Functions
# ===========================================================================

def run_check(
    row: Union[pd.Series, Dict[str, Any]],
    image: Optional[np.ndarray] = None,
) -> InspectionItemResult:
    """Evaluate a single checklist row according to explainable rules.

    Parameters
    ----------
    row : Union[pd.Series, Dict[str, Any]]
        Checklist record containing check_id, check_name, expected_value, etc.
    image : Optional[np.ndarray]
        Cabinet photo if available for visual extraction.

    Returns
    -------
    InspectionItemResult
        Object containing check_id, check_name, expected, found, status,
        reason, evidence_box, and source_type.
    """
    get = (lambda k, d=None: row.get(k, d)) if isinstance(row, dict) else (lambda k, d=None: row.get(k, d))

    check_id = str(get("check_id", "CHK-UNKNOWN")).strip()
    check_name = str(get("check_name", "Unnamed Check")).strip()
    expected = str(get("expected_value", get("expected", "N/A"))).strip()

    raw_found = get("found_value", get("found", get("demo_found_value")))
    found = str(raw_found).strip() if pd.notna(raw_found) and str(raw_found).strip() != "" else None

    notes = str(get("notes", get("reason", ""))).strip()
    from src.eagle_eye.input_processing.file_loader import parse_evidence_box, parse_demo_status
    evidence_box = parse_evidence_box(get("evidence_box", get("evidence_bbox")))

    # Check for explicit source_type or rule overrides
    source_type_col = str(get("source_type", "")).strip().lower()
    has_demo_status = "demo_status" in row and pd.notna(get("demo_status")) and str(get("demo_status")).strip() != ""
    rule_type = str(get("check_type", get("rule_type", ""))).strip().lower()

    # Case A: Seeded Demo Record (from sample checklist)
    if has_demo_status or source_type_col == "demo_seeded":
        status = parse_demo_status(get("demo_status", get("status")))
        found_val = found if found is not None else (expected if status == CheckStatus.PASS else "N/A")
        reason = notes if notes else f"Seeded demo inspection result: {status.value}"
        src_type = "demo_seeded"

        return InspectionItemResult(
            check_id=check_id,
            check_name=check_name,
            expected_value=expected,
            found_value=found_val,
            status=status,
            notes=reason,
            evidence_bbox=evidence_box,
            source_type=src_type,
        )

    # Case B: Explicit Rule Evaluation
    if rule_type in {"exact_label_match", "label_match"} or "label" in check_name.lower():
        # Try OCR if image and bbox provided and found is not already set
        if found is None and image is not None and evidence_box is not None:
            from src.eagle_eye.vision.image_processor import try_ocr_region
            ocr_text = try_ocr_region(image, evidence_box)
            if ocr_text:
                found = ocr_text

        status, reason, found_val, src_type = evaluate_exact_label_match(expected, found)

    elif rule_type in {"order_sequence", "order_mismatch"} or "order" in check_name.lower() or "sequence" in check_name.lower():
        status, reason, found_val, src_type = evaluate_order_sequence_check(expected, found)

    elif rule_type in {"missing_part", "component_presence"} or "presence" in check_name.lower():
        detector_miss = bool(get("detector_miss", False))
        status, reason, found_val, src_type = evaluate_missing_part_check(
            expected, found, is_seeded_demo=False, detector_miss=detector_miss
        )

    elif rule_type in {"manual_check", "uncertain"}:
        status, reason, found_val, src_type = evaluate_manual_glare_case(expected, found, notes or "Visual ambiguity")

    elif rule_type in {"indicator_status", "indicator"} or "led" in check_name.lower() or "indicator" in check_name.lower():
        status, reason, found_val, src_type = evaluate_indicator_state(expected, found)

    elif found is not None and found not in {"Pending Verification", "N/A", ""}:
        # General string match
        if found.strip().lower() == expected.strip().lower():
            status = CheckStatus.PASS
            reason = f"Verified: expected '{expected}' matches found '{found}'."
            found_val = found
            src_type = "automatic"
        elif any(k in found.lower() for k in ["missing", "disconnected", "broken"]):
            status = CheckStatus.FAIL
            reason = f"Discrepancy confirmed: {found}."
            found_val = found
            src_type = "automatic"
        else:
            status = CheckStatus.MANUAL_CHECK
            reason = f"Observed '{found}' differs from expected '{expected}'; manual review required."
            found_val = found
            src_type = "automatic"

    else:
        # Baseline: Pending automated inspection -> Needs Manual Check
        status = CheckStatus.MANUAL_CHECK
        found_val = "Pending Verification"
        reason = f"Pending visual verification. {notes}".strip()
        src_type = "automatic"

    if notes and notes not in reason:
        reason = f"{reason} | {notes}".strip(" |")

    return InspectionItemResult(
        check_id=check_id,
        check_name=check_name,
        expected_value=expected,
        found_value=found_val,
        status=status,
        notes=reason,
        evidence_bbox=evidence_box,
        source_type=src_type,
    )


def run_checks(
    checklist_rows: Union[pd.DataFrame, List[Dict[str, Any]]],
    image: Optional[np.ndarray] = None,
) -> List[InspectionItemResult]:
    """Execute rule-based checks across multiple checklist rows.

    Parameters
    ----------
    checklist_rows : Union[pd.DataFrame, List[Dict[str, Any]]]
        Checklist entries to evaluate.
    image : Optional[np.ndarray]
        Cabinet photo if available.

    Returns
    -------
    List[InspectionItemResult]
        Evaluated item results with sequential marker IDs.
    """
    if isinstance(checklist_rows, pd.DataFrame):
        rows = [row.to_dict() for _, row in checklist_rows.iterrows()]
    else:
        rows = checklist_rows

    results: List[InspectionItemResult] = []
    marker_counter = 1

    for row_dict in rows:
        item = run_check(row_dict, image=image)
        # Assign marker ID if item has a bounding box or flagged status
        if item.evidence_bbox is not None or item.status != CheckStatus.PASS:
            item.marker_id = marker_counter
            marker_counter += 1
        results.append(item)

    return results


def run_inspection(
    image: Optional[np.ndarray],
    checklist_df: Union[pd.DataFrame, List[Dict[str, Any]]],
) -> InspectionSummary:
    """Execute quality inspection checks against checklist items.

    Parameters
    ----------
    image : Optional[np.ndarray]
        OpenCV BGR image of the cabinet, if uploaded.
    checklist_df : Union[pd.DataFrame, List[Dict[str, Any]]]
        DataFrame or list of row dicts containing checklist items.

    Returns
    -------
    InspectionSummary
        Inspection summary populated with structured results.
    """
    if isinstance(checklist_df, pd.DataFrame):
        has_demo_columns = any(col in checklist_df.columns for col in ["demo_status", "demo_found_value", "evidence_box"])
        # If standard demo checklist without custom rule column, use file_loader conversion to preserve exact demo mappings
        if has_demo_columns and not any(col in checklist_df.columns for col in ["check_type", "rule_type"]):
            from src.eagle_eye.input_processing.file_loader import checklist_to_inspection_results
            items = checklist_to_inspection_results(checklist_df)
            return InspectionSummary.from_items(items)

    items = run_checks(checklist_df, image=image)
    return InspectionSummary.from_items(items)


# ===========================================================================
# Re-Inspection / Re-check Rerun Function
# ===========================================================================

def recheck_inspection(
    original_summary: InspectionSummary,
    target_check_id: str,
    fixed_found_value: Optional[str] = None,
    updated_checklist_df: Optional[pd.DataFrame] = None,
    updated_image: Optional[np.ndarray] = None,
    resolution_reason: str = "Resolved during re-inspection.",
) -> InspectionSummary:
    """Re-check a previously failing item after a simulated or real fix.

    Demonstrates closed-loop re-inspection workflow where a defect transitions
    from FAIL to PASS.

    Parameters
    ----------
    original_summary : InspectionSummary
        Previous inspection results.
    target_check_id : str
        ID of the item that was corrected (e.g. 'CHK-002').
    fixed_found_value : Optional[str]
        Verified corrected finding. Defaults to expected_value.
    updated_checklist_df : Optional[pd.DataFrame]
        If an entire updated checklist is provided, re-runs full inspection.
    updated_image : Optional[np.ndarray]
        Updated photo if provided.
    resolution_reason : str
        Audit explanation of the resolution.

    Returns
    -------
    InspectionSummary
        Updated summary showing resolved status.
    """
    if updated_checklist_df is not None:
        return run_inspection(image=updated_image, checklist_df=updated_checklist_df)

    new_items = deepcopy(original_summary.items)

    for item in new_items:
        if item.check_id == target_check_id:
            item.status = CheckStatus.PASS
            item.found_value = fixed_found_value if fixed_found_value is not None else item.expected_value
            item.notes = f"{item.notes} | [RE-INSPECTED: {resolution_reason}]".strip(" |")

    return InspectionSummary.from_items(new_items)
