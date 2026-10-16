"""Inspection report generator for exporting inspection outcomes.

Owned by: MEMBER 4 (UI, Reporting and Integration)
"""

from pathlib import Path
from typing import Union
import pandas as pd

from src.eagle_eye.inspection.result_models import InspectionSummary


def generate_summary_dataframe(summary: InspectionSummary) -> pd.DataFrame:
    """Transform an InspectionSummary into a displayable pandas DataFrame.

    Parameters
    ----------
    summary : InspectionSummary
        The completed inspection summary.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe with check results.
    """
    records = []
    for item in summary.items:
        records.append({
            "Check ID": item.check_id,
            "Check Description": item.check_name,
            "Expected": item.expected_value,
            "Found": item.found_value if item.found_value is not None else "N/A",
            "Status": item.status.value,
            "Source": getattr(item, "source_type", "demo_seeded"),
            "Notes": item.notes,
        })
    return pd.DataFrame(records)


def export_report_csv(summary: InspectionSummary, output_path: Union[str, Path]) -> str:
    """Save an inspection summary as a CSV report.

    Parameters
    ----------
    summary : InspectionSummary
        The inspection summary to export.
    output_path : Union[str, Path]
        Target file path for the CSV report.

    Returns
    -------
    str
        Path to the saved report.
    """
    df = generate_summary_dataframe(summary)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def export_report_markdown(summary: InspectionSummary) -> str:
    """Generate a clean Markdown summary of the inspection results.

    Parameters
    ----------
    summary : InspectionSummary
        The inspection summary to format.

    Returns
    -------
    str
        Markdown report string.
    """
    overall = "PASSED" if summary.is_overall_pass() else "ACTION REQUIRED"
    md = [
        "# EAGLE EYE Inspection Report",
        "",
        f"- **Overall Status:** {overall}",
        f"- **Total Checks:** {summary.total_checks}",
        f"- **Passed:** {summary.passed_count}",
        f"- **Failed:** {summary.failed_count}",
        f"- **Needs Manual Check:** {summary.manual_check_count}",
        "",
        "## Detailed Checklist Results",
        "",
        "| Check ID | Item | Expected | Found | Status | Source | Notes |",
        "|---|---|---|---|---|---|---|",
    ]

    for item in summary.items:
        source_label = getattr(item, "source_type", "demo_seeded")
        md.append(
            f"| {item.check_id} | {item.check_name} | {item.expected_value} | "
            f"{item.found_value or 'N/A'} | {item.status.value} | {source_label} | {item.notes} |"
        )

    return "\n".join(md)
