"""Data structures and status enumerations for inspection results.

Owned by: MEMBER 3 (Inspection Logic)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple, Dict, Any, Union


class CheckStatus(str, Enum):
    """Quality check statuses supported by Eagle Eye."""
    PASS = "PASS"
    FAIL = "FAIL"
    MANUAL_CHECK = "Needs Manual Check"


@dataclass
class InspectionItemResult:
    """Inspection outcome for a single checklist line item."""
    check_id: str
    check_name: str
    expected_value: str
    found_value: Optional[str] = None
    status: CheckStatus = CheckStatus.MANUAL_CHECK
    notes: str = ""
    evidence_bbox: Optional[Tuple[int, int, int, int]] = None
    marker_id: Optional[int] = None
    source_type: str = "demo_seeded"

    def __init__(
        self,
        check_id: str,
        check_name: str,
        expected_value: str = "",
        found_value: Optional[str] = None,
        status: Union[CheckStatus, str] = CheckStatus.MANUAL_CHECK,
        notes: str = "",
        evidence_bbox: Optional[Tuple[int, int, int, int]] = None,
        marker_id: Optional[int] = None,
        source_type: str = "demo_seeded",
        *,
        expected: Optional[str] = None,
        found: Optional[str] = None,
        reason: Optional[str] = None,
        evidence_box: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.check_id = str(check_id)
        self.check_name = str(check_name)
        self.expected_value = str(expected if expected is not None else expected_value)
        self.found_value = str(found) if found is not None else (str(found_value) if found_value is not None else None)
        if isinstance(status, CheckStatus):
            self.status = status
        else:
            norm_s = str(status).strip()
            if norm_s.upper() == "PASS":
                self.status = CheckStatus.PASS
            elif norm_s.upper() == "FAIL":
                self.status = CheckStatus.FAIL
            else:
                self.status = CheckStatus.MANUAL_CHECK
        self.notes = str(reason if reason is not None else notes)
        self.evidence_bbox = evidence_box if evidence_box is not None else evidence_bbox
        self.marker_id = marker_id
        self.source_type = str(source_type)

    @property
    def expected(self) -> str:
        """Alias for expected_value."""
        return self.expected_value

    @property
    def found(self) -> Optional[str]:
        """Alias for found_value."""
        return self.found_value

    @property
    def reason(self) -> str:
        """Alias for notes."""
        return self.notes

    @property
    def evidence_box(self) -> Optional[Tuple[int, int, int, int]]:
        """Alias for evidence_bbox."""
        return self.evidence_bbox

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like subscripting for compatibility."""
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the item result to a serializable dictionary."""
        return {
            "check_id": self.check_id,
            "check_name": self.check_name,
            "expected_value": self.expected_value,
            "expected": self.expected_value,
            "found_value": self.found_value if self.found_value is not None else "N/A",
            "found": self.found_value if self.found_value is not None else "N/A",
            "status": self.status.value,
            "notes": self.notes,
            "reason": self.notes,
            "evidence_bbox": self.evidence_bbox,
            "evidence_box": self.evidence_bbox,
            "marker_id": self.marker_id,
            "source_type": self.source_type,
            "has_evidence": self.evidence_bbox is not None or self.marker_id is not None,
        }


@dataclass
class InspectionSummary:
    """Aggregated summary of an inspection run."""
    total_checks: int
    passed_count: int
    failed_count: int
    manual_check_count: int
    items: List[InspectionItemResult] = field(default_factory=list)

    @classmethod
    def from_items(cls, items: List[InspectionItemResult]) -> "InspectionSummary":
        """Construct an InspectionSummary by aggregating item results."""
        total = len(items)
        passed = sum(1 for item in items if item.status == CheckStatus.PASS)
        failed = sum(1 for item in items if item.status == CheckStatus.FAIL)
        manual = sum(1 for item in items if item.status == CheckStatus.MANUAL_CHECK)
        return cls(
            total_checks=total,
            passed_count=passed,
            failed_count=failed,
            manual_check_count=manual,
            items=items,
        )

    def is_overall_pass(self) -> bool:
        """Return True if all items passed with zero failures or manual checks."""
        return self.total_checks > 0 and self.passed_count == self.total_checks

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary format for reporting."""
        return {
            "total_checks": self.total_checks,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "manual_check_count": self.manual_check_count,
            "overall_status": "PASS" if self.is_overall_pass() else "ACTION_REQUIRED",
            "items": [item.to_dict() for item in self.items],
        }
