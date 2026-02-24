#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

IGNORE_VALUE = 255
VALID_MASK_VALUES = {0, 1, 2, 3, 4, 5, IGNORE_VALUE}

# Canonical class IDs requested by the user.
CLASS_ID_TO_NAME = {
    1: "Epithelial tissue",
    2: "Slough",
    3: "Granulation tissue",
    4: "Necrotic tissue",
    5: "Other",
}

# Visualization colors: distinct palette to avoid confusion.
# Epithelial=green, Slough=yellow, Granulation=magenta, Necrotic=dark orange, Other=cyan, ignore=light gray.
# BGR for OpenCV (e.g. cv2.imwrite, overlay).
CLASS_COLORS_BGR = {
    0: (0, 0, 0),           # background black
    1: (0, 255, 0),         # Epithelial tissue green
    2: (0, 255, 255),       # Slough yellow
    3: (255, 0, 255),       # Granulation tissue magenta
    4: (49, 144, 250),      # Necrotic tissue dark orange #FA9031
    5: (255, 255, 0),       # Other cyan
    255: (200, 200, 200),   # ignore light gray
}
# RGB for saved PNG / matplotlib (R, G, B).
CLASS_COLORS_RGB = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (255, 255, 0),
    3: (255, 0, 255),
    4: (250, 144, 49),
    5: (0, 255, 255),
    255: (200, 200, 200),
}

# Accept aliases from historical label configs/exports.
CLASS_NAME_TO_ID = {
    "epithelial tissue": 1,
    "epithelial_tissue": 1,
    "epithelization": 1,
    "epitelization": 1,
    "slough": 2,
    "fibrin": 2,
    "granulation tissue": 3,
    "granulation_tissue": 3,
    "granulation": 3,
    "necrotic tissue": 4,
    "necrotic_tissue": 4,
    "necrotic": 4,
    "other": 5,
}

WOUND_LABEL_ALIASES = {"wound_outline", "wound"}

# Golden-set patient IDs (used for val split and visualization exclusion by default).
GOLDEN_PATIENTS = frozenset({
    "Patient_1", "Patient_5", "Patient_6", "Patient_11", "Patient_13",
    "Patient_20", "Patient_28", "Patient_33", "Patient_39",
})


def normalize_label(label: str) -> str:
    return label.strip().lower()


def tissue_label_to_id(label: str) -> int | None:
    return CLASS_NAME_TO_ID.get(normalize_label(label))


def parse_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_csv_arg(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def infer_patient_id(task_data: dict) -> str | None:
    patient_id = task_data.get("patient_id")
    if isinstance(patient_id, str) and patient_id:
        return patient_id
    image_value = str(task_data.get("image", ""))
    match = re.search(r"(Patient_\\d+)", image_value)
    if match:
        return match.group(1)
    return None


def infer_doctor_id(image_value: str) -> str | None:
    match = re.search(r"(?:P1_wound_|P2_tissue_)(D\\d+)", image_value)
    if match:
        return match.group(1)
    return None


def image_relpath_from_value(image_value: str) -> str:
    """
    Convert data.image into a relative path under Dataset_evolution_wounds_VR.
    """
    if not image_value:
        return ""
    parsed = urlparse(image_value)
    raw_path = parsed.path if parsed.scheme else image_value
    marker = "Dataset_evolution_wounds_VR/"
    idx = raw_path.find(marker)
    if idx >= 0:
        return raw_path[idx + len(marker) :].lstrip("/")
    return os.path.basename(raw_path)


def canonical_image_key(image_relpath: str) -> str:
    """
    Canonical key used to merge duplicated golden-set images across doctor folders.

    Examples:
    - P1_wound_D1/Patient_11/P11T_2.jpg -> Patient_11/P11T_2.jpg
    - P2_tissue_D3/Patient_11/P11T_2.jpg -> Patient_11/P11T_2.jpg
    """
    rel = (image_relpath or "").lstrip("/")
    parts = rel.split("/")
    if len(parts) >= 3 and re.match(r"^P[12]_(?:wound|tissue)_D\d+$", parts[0]):
        return "/".join(parts[1:])
    return rel


def image_key_from_value(image_value: str) -> str:
    """
    Backward-compatible helper: canonical key from data.image.
    """
    return canonical_image_key(image_relpath_from_value(image_value))


def find_image_path(
    image_key: str,
    image_name: str | None,
    images_root: str | Path,
    basename_cache: dict[str, list[str]] | None = None,
) -> str | None:
    root = Path(images_root).resolve()
    candidate = root / image_key
    if image_key and candidate.exists():
        return str(candidate)

    if not image_name:
        return None

    if basename_cache is not None:
        paths = basename_cache.get(image_name, [])
        if len(paths) == 1:
            return paths[0]
        if len(paths) > 1:
            # If ambiguous, try to find a path containing the patient folder from image_key.
            for p in paths:
                if image_key and image_key in p:
                    return p
            return None

    # Fallback linear scan.
    matches = list(root.rglob(image_name))
    if len(matches) == 1:
        return str(matches[0])
    return None


def build_basename_cache(images_root: str | Path) -> dict[str, list[str]]:
    root = Path(images_root).resolve()
    cache: dict[str, list[str]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            continue
        cache.setdefault(path.name, []).append(str(path))
    return cache


@dataclass
class SummaryCounter:
    records: int = 0
    polygons_total: int = 0
    skipped_no_annotations: int = 0
    skipped_no_tissue: int = 0
    skipped_missing_image: int = 0
