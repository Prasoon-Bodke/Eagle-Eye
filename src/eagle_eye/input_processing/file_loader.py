"""File loading and validation utilities for images and checklists.

Owned by: MEMBER 1 (Input Processing)
"""

import io
from pathlib import Path
from typing import Union, BinaryIO, Optional, Tuple, List, Any
import cv2
import numpy as np
import pandas as pd

from config.settings import (
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_CHECKLIST_EXTENSIONS,
    REQUIRED_CHECKLIST_COLUMNS,
    DEFAULT_CHECKLIST_PATH,
    DEFAULT_SAMPLE_IMAGE_PATH,
    DEFAULT_DEMO_CHECKLIST_PATH,
)
from src.eagle_eye.inspection.result_models import (
    CheckStatus,
    InspectionItemResult,
)


def is_valid_image_extension(filename: str) -> bool:
    """Check if the filename has a supported image extension (.jpg, .jpeg, .png)."""
    if not filename:
        return False
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def is_valid_checklist_extension(filename: str) -> bool:
    """Check if the filename has a supported checklist extension (.csv, .xlsx, .xls)."""
    if not filename:
        return False
    ext = Path(filename).suffix.lower()
    return ext in SUPPORTED_CHECKLIST_EXTENSIONS


def parse_evidence_box(val: Any) -> Optional[Tuple[int, int, int, int]]:
    """Parse bounding box coordinates from a string or iterable.

    Supports formats:
    - "160,140,360,320"
    - "[160, 140, 360, 320]"
    - "(160, 140, 360, 320)"
    - [160, 140, 360, 320] or (160, 140, 360, 320)

    Returns (x1, y1, x2, y2) or None if invalid or absent.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None

    coords = None
    if isinstance(val, (list, tuple)) and len(val) == 4:
        coords = val
    elif isinstance(val, str):
        cleaned = val.strip().strip("[]()").strip()
        if not cleaned:
            return None
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        if len(parts) == 4:
            coords = parts

    if coords is not None:
        try:
            x1, y1, x2, y2 = [int(float(c)) for c in coords]
            if x1 < x2 and y1 < y2 and min(x1, y1, x2, y2) >= 0:
                return (x1, y1, x2, y2)
        except (ValueError, TypeError):
            return None

    return None


def parse_demo_status(val: Any) -> CheckStatus:
    """Parse raw status string into standardized CheckStatus enum.

    Defaults safely to CheckStatus.MANUAL_CHECK if unclear, unparseable, or empty.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return CheckStatus.MANUAL_CHECK

    s = str(val).strip().upper()
    if s == "PASS":
        return CheckStatus.PASS
    elif s == "FAIL":
        return CheckStatus.FAIL
    elif s in {"NEEDS MANUAL CHECK", "MANUAL_CHECK", "MANUAL", "REVIEW"}:
        return CheckStatus.MANUAL_CHECK
    else:
        # Default to manual check when evidence is ambiguous
        return CheckStatus.MANUAL_CHECK


def load_checklist(file_source: Union[str, Path, BinaryIO]) -> pd.DataFrame:
    """Load an inspection checklist from a CSV or Excel file.

    Parameters
    ----------
    file_source : str, Path, or file-like buffer
        The path or uploaded file buffer containing the checklist.

    Returns
    -------
    pd.DataFrame
        Loaded checklist dataframe with validated required columns.

    Raises
    ------
    FileNotFoundError
        If a file path is provided that does not exist.
    ValueError
        If the file format is unsupported, empty, or required columns are missing.
    """
    ext = ""
    is_path = isinstance(file_source, (str, Path))

    if is_path:
        path = Path(file_source)
        if not path.exists():
            raise FileNotFoundError(f"Checklist file not found at: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Checklist file '{path.name}' is empty (0 bytes).")
        ext = path.suffix.lower()
        if not is_valid_checklist_extension(path.name):
            raise ValueError(
                f"Unsupported checklist format: '{ext}'. "
                f"Expected one of: {', '.join(sorted(SUPPORTED_CHECKLIST_EXTENSIONS))}."
            )
        try:
            if ext == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
        except Exception as e:
            raise ValueError(f"Failed to parse checklist file '{path.name}': {e}")
    else:
        # File-like object (e.g. Streamlit UploadedFile, BytesIO, StringIO)
        filename = getattr(file_source, "name", "")
        if filename:
            ext = Path(filename).suffix.lower()
            if not is_valid_checklist_extension(filename):
                raise ValueError(
                    f"Unsupported checklist format: '{ext}'. "
                    f"Expected one of: {', '.join(sorted(SUPPORTED_CHECKLIST_EXTENSIONS))}."
                )

        # Check buffer position and size if accessible
        if hasattr(file_source, "seek"):
            file_source.seek(0)

        try:
            if ext in {".xlsx", ".xls"}:
                df = pd.read_excel(file_source)
            else:
                df = pd.read_csv(file_source)
        except pd.errors.EmptyDataError:
            raise ValueError("Checklist file is empty or contains no readable data.")
        except Exception as e:
            raise ValueError(f"Failed to parse checklist content: {e}")

    # Validate non-empty DataFrame
    if df.empty:
        raise ValueError("Checklist contains no data rows.")

    # Validate required columns
    missing_cols = [col for col in REQUIRED_CHECKLIST_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Checklist is missing required column(s): {missing_cols}. "
            f"Required columns are: {REQUIRED_CHECKLIST_COLUMNS}"
        )

    # Clean whitespace in string columns and replace NaN with empty string
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def validate_checklist_file(
    file_source: Union[str, Path, BinaryIO],
    filename: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """Safely validate and parse a checklist file, catching all errors.

    Returns
    -------
    Tuple[bool, Optional[str], Optional[pd.DataFrame]]
        (is_valid, error_message, parsed_dataframe)
    """
    fname = filename or getattr(file_source, "name", str(file_source) if isinstance(file_source, (str, Path)) else "")
    if fname and not is_valid_checklist_extension(fname):
        ext = Path(fname).suffix.lower()
        return (
            False,
            f"Invalid file format '{ext}'. Checklists must be CSV or Excel ({', '.join(sorted(SUPPORTED_CHECKLIST_EXTENSIONS))}).",
            None,
        )

    try:
        df = load_checklist(file_source)
        return (True, None, df)
    except FileNotFoundError as e:
        return (False, str(e), None)
    except ValueError as e:
        return (False, str(e), None)
    except Exception as e:
        return (False, f"Unexpected error loading checklist: {e}", None)


def load_cabinet_image(
    file_source: Union[str, Path, BinaryIO, bytes],
    filename: Optional[str] = None,
) -> np.ndarray:
    """Decode and validate a cabinet photo into an OpenCV BGR image array.

    Parameters
    ----------
    file_source : str, Path, BinaryIO, or bytes
        File path, raw bytes, or uploaded buffer.
    filename : Optional[str]
        Optional filename to validate extension when using a raw buffer.

    Returns
    -------
    np.ndarray
        Decoded image in OpenCV BGR format.

    Raises
    ------
    FileNotFoundError
        If a file path is provided that does not exist.
    ValueError
        If the file has an unsupported extension or cannot be decoded.
    """
    raw_bytes: bytes

    if isinstance(file_source, (str, Path)):
        path = Path(file_source)
        if not path.exists():
            raise FileNotFoundError(f"Cabinet image file not found at: {path}")
        if not is_valid_image_extension(path.name):
            raise ValueError(
                f"Unsupported image extension '{path.suffix.lower()}'. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}."
            )
        if path.stat().st_size == 0:
            raise ValueError(f"Cabinet image file '{path.name}' is empty (0 bytes).")
        raw_bytes = path.read_bytes()
    elif isinstance(file_source, (bytes, bytearray)):
        raw_bytes = bytes(file_source)
        if filename and not is_valid_image_extension(filename):
            raise ValueError(
                f"Unsupported image extension for '{filename}'. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}."
            )
    else:
        # Buffer-like (e.g. Streamlit UploadedFile)
        buf_name = filename or getattr(file_source, "name", "")
        if buf_name and not is_valid_image_extension(buf_name):
            raise ValueError(
                f"Unsupported image extension for '{buf_name}'. "
                f"Supported extensions: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}."
            )
        if hasattr(file_source, "seek"):
            file_source.seek(0)
        raw_bytes = file_source.read()

    if not raw_bytes or len(raw_bytes) == 0:
        raise ValueError("Image data is empty (0 bytes).")

    nparr = np.frombuffer(raw_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise ValueError(
            "Failed to decode cabinet image. The file may be corrupted, truncated, or not a valid image format."
        )

    return image


def validate_image_file(
    file_source: Union[str, Path, BinaryIO, bytes],
    filename: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[np.ndarray]]:
    """Safely validate and decode an uploaded cabinet image.

    Returns
    -------
    Tuple[bool, Optional[str], Optional[np.ndarray]]
        (is_valid, error_message, decoded_image)
    """
    fname = filename or getattr(file_source, "name", str(file_source) if isinstance(file_source, (str, Path)) else "")
    if fname and not is_valid_image_extension(fname):
        ext = Path(fname).suffix.lower()
        return (
            False,
            f"Invalid image format '{ext}'. Uploaded cabinet photos must be JPG or PNG ({', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}).",
            None,
        )

    try:
        img = load_cabinet_image(file_source, filename=fname)
        return (True, None, img)
    except FileNotFoundError as e:
        return (False, str(e), None)
    except ValueError as e:
        return (False, str(e), None)
    except Exception as e:
        return (False, f"Unexpected error reading image: {e}", None)


def get_sample_image_path() -> Path:
    """Return the filesystem path to the sample cabinet image."""
    return DEFAULT_SAMPLE_IMAGE_PATH


def get_sample_checklist_path() -> Path:
    """Return the filesystem path to the sample demo checklist."""
    if DEFAULT_DEMO_CHECKLIST_PATH.exists():
        return DEFAULT_DEMO_CHECKLIST_PATH
    return DEFAULT_CHECKLIST_PATH


def load_sample_demo_data() -> Tuple[np.ndarray, pd.DataFrame]:
    """Load both the sample cabinet photo and the demo checklist.

    Returns
    -------
    Tuple[np.ndarray, pd.DataFrame]
        (sample_image_bgr, sample_checklist_df)

    Raises
    ------
    FileNotFoundError
        If required sample files are missing.
    """
    img_path = get_sample_image_path()
    chk_path = get_sample_checklist_path()

    if not img_path.exists():
        raise FileNotFoundError(f"Sample cabinet image missing at: {img_path}")
    if not chk_path.exists():
        raise FileNotFoundError(f"Sample checklist missing at: {chk_path}")

    image = load_cabinet_image(img_path)
    checklist_df = load_checklist(chk_path)
    return image, checklist_df


def checklist_to_inspection_results(df: pd.DataFrame) -> List[InspectionItemResult]:
    """Convert a validated checklist DataFrame into a list of InspectionItemResults.

    Extracts demo columns (demo_found_value, demo_status, evidence_box) if present,
    or sets default baseline values ("Pending Verification", CheckStatus.MANUAL_CHECK).
    Assigns sequential marker IDs for items that have bounding boxes or flagged statuses.
    """
    items: List[InspectionItemResult] = []
    marker_counter = 1

    for idx, row in df.iterrows():
        check_id = str(row.get("check_id", f"CHK-{idx+1:03d}")).strip()
        check_name = str(row.get("check_name", "Unnamed Check")).strip()
        expected = str(row.get("expected_value", "N/A")).strip()
        notes = str(row.get("notes", "")).strip()
        comp_type = str(row.get("component_type", "")).strip()

        # Demo fields
        has_demo_status = "demo_status" in row and pd.notna(row.get("demo_status")) and str(row.get("demo_status")).strip() != ""
        if has_demo_status:
            status = parse_demo_status(row.get("demo_status"))
        else:
            status = CheckStatus.MANUAL_CHECK

        demo_found = row.get("demo_found_value") if "demo_found_value" in row else row.get("found_value")
        if pd.notna(demo_found) and str(demo_found).strip() != "":
            found_val = str(demo_found).strip()
        else:
            found_val = "Pending Verification" if not has_demo_status else expected

        # Evidence bbox
        evidence_bbox = None
        if "evidence_box" in row and pd.notna(row.get("evidence_box")):
            evidence_bbox = parse_evidence_box(row.get("evidence_box"))

        # Assign marker ID if bbox is present or if flagged
        marker_id = None
        if evidence_bbox is not None or status != CheckStatus.PASS:
            marker_id = marker_counter
            marker_counter += 1

        # Combine notes
        combined_notes = notes
        # Determine source_type: automatic or demo_seeded
        if "source_type" in row and pd.notna(row.get("source_type")) and str(row.get("source_type")).strip():
            src_type = str(row.get("source_type")).strip()
        elif has_demo_status or "demo_found_value" in row:
            src_type = "demo_seeded"
        else:
            src_type = "automatic"

        item = InspectionItemResult(
            check_id=check_id,
            check_name=check_name,
            expected_value=expected,
            found_value=found_val,
            status=status,
            notes=combined_notes,
            evidence_bbox=evidence_bbox,
            marker_id=marker_id,
            source_type=src_type,
        )
        items.append(item)

    return items
