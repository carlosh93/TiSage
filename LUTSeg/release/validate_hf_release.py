#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image


EXPECTED_CLINICIANS = ("user_4", "user_5", "user_6", "user_7", "user_9")
EXPECTED_COLLECTIONS = {
    "Images": 141,
    "Masks": 141,
    "Masks_RGB": 141,
    "Wound_Masks": 141,
}
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}
ABSOLUTE_PATH = re.compile(r"/(?:home|Users|scratch|gpfs|ibex|data/projects)/")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
HF_SECRET = re.compile(r"(?:HF_TOKEN\s*=|hf_[A-Za-z0-9]{20,})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a staged LUTSeg release")
    parser.add_argument("release", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
    return records


def safe_path(root: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"expected relative path string, got {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe release path: {value}")
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise ValueError(f"missing release file: {value}")
    return path


def count_files(path: Path, suffixes: set[str]) -> int:
    return sum(
        1
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in suffixes
    )


def image_array(path: Path) -> tuple[np.ndarray, str, dict[str, object], object]:
    with Image.open(path) as image:
        array = np.asarray(image)
        return array, image.mode, dict(image.info), image.getexif()


def validate_main_records(root: Path) -> None:
    records = read_jsonl(root / "metadata.jsonl")
    if len(records) != 141:
        raise ValueError(f"expected 141 metadata rows, found {len(records)}")
    split_counts = Counter(record.get("split") for record in records)
    if split_counts != {"train": 111, "validation": 30}:
        raise ValueError(f"unexpected split counts: {dict(split_counts)}")
    patients = {record.get("patient_id") for record in records}
    if len(patients) != 39:
        raise ValueError(f"expected 39 patient codes, found {len(patients)}")

    train_patients = {
        record["patient_id"] for record in records if record["split"] == "train"
    }
    validation_patients = {
        record["patient_id"] for record in records if record["split"] == "validation"
    }
    overlap = train_patients & validation_patients
    if overlap:
        raise ValueError(f"patient leakage between splits: {sorted(overlap)}")

    required_paths = (
        "image_file_name",
        "mask_file_name",
        "mask_rgb_file_name",
        "wound_mask_file_name",
    )
    seen_images = set()
    for record in records:
        paths = {key: safe_path(root, record.get(key)) for key in required_paths}
        if record["image_file_name"] in seen_images:
            raise ValueError(f"duplicate metadata image: {record['image_file_name']}")
        seen_images.add(record["image_file_name"])

        image, image_mode, image_info, image_exif = image_array(paths["image_file_name"])
        mask, mask_mode, mask_info, mask_exif = image_array(paths["mask_file_name"])
        mask_rgb, rgb_mode, rgb_info, rgb_exif = image_array(paths["mask_rgb_file_name"])
        wound, wound_mode, wound_info, wound_exif = image_array(
            paths["wound_mask_file_name"]
        )
        if image_mode != "RGB" or rgb_mode != "RGB":
            raise ValueError(f"unexpected RGB mode for {record['image_file_name']}")
        if mask_mode != "L" or wound_mode != "L":
            raise ValueError(f"unexpected indexed-mask mode for {record['image_file_name']}")
        if image.shape[:2] != mask.shape or mask.shape != wound.shape:
            raise ValueError(f"image/mask size mismatch for {record['image_file_name']}")
        if mask_rgb.shape[:2] != mask.shape:
            raise ValueError(f"RGB mask size mismatch for {record['image_file_name']}")

        mask_values = {int(value) for value in np.unique(mask)}
        wound_values = {int(value) for value in np.unique(wound)}
        if not mask_values <= {0, 1, 2, 3, 4, 5}:
            raise ValueError(f"invalid tissue labels in {record['mask_file_name']}")
        if not wound_values <= {0, 255}:
            raise ValueError(f"invalid wound labels in {record['wound_mask_file_name']}")
        if sorted(mask_values) != record.get("classes_present"):
            raise ValueError(f"class metadata mismatch for {record['mask_file_name']}")

        forbidden_info = {"exif", "xmp", "photoshop", "comment"}
        if forbidden_info & set(image_info) or image_exif:
            raise ValueError(f"embedded image metadata remains in {record['image_file_name']}")
        for path_key, info, exif in (
            ("mask_file_name", mask_info, mask_exif),
            ("mask_rgb_file_name", rgb_info, rgb_exif),
            ("wound_mask_file_name", wound_info, wound_exif),
        ):
            if info or exif:
                raise ValueError(f"embedded mask metadata remains in {record[path_key]}")

    expected_pairs = {
        split: {
            tuple(line.split())
            for line in (root / filename).read_text().splitlines()
            if line.strip()
        }
        for split, filename in (("train", "train.txt"), ("validation", "val.txt"))
    }
    metadata_pairs = {
        split: {
            (record["image_file_name"], record["mask_file_name"])
            for record in records
            if record["split"] == split
        }
        for split in ("train", "validation")
    }
    if expected_pairs != metadata_pairs:
        raise ValueError("train/validation text files do not match metadata.jsonl")


def validate_gold_standard(root: Path) -> None:
    gold_root = root / "gold_standard"
    records = read_jsonl(gold_root / "manifest.jsonl")
    if len(records) != 46:
        raise ValueError(f"expected 46 gold-standard rows, found {len(records)}")
    patients = {record.get("patient_id") for record in records}
    if len(patients) != 9:
        raise ValueError(f"expected 9 gold-standard patient codes, found {len(patients)}")

    expected_tissue = 46 * len(EXPECTED_CLINICIANS)
    actual_all = count_files(gold_root / "masks_by_clinician", {".png"})
    if actual_all != expected_tissue * 2:
        raise ValueError(
            f"expected {expected_tissue} tissue and {expected_tissue} wound expert masks, "
            f"found {actual_all} total"
        )

    referenced_masks = set()
    for record in records:
        image_path = safe_path(root, record.get("image_file_name"))
        with Image.open(image_path) as image:
            image_size = image.size
        for clinician in EXPECTED_CLINICIANS:
            for mask_type, allowed in (
                ("tissue", {0, 1, 2, 3, 4, 5}),
                ("wound", {0, 255}),
            ):
                key = f"{clinician}_{mask_type}_mask"
                mask_path = safe_path(root, record.get(key))
                relative_mask = mask_path.relative_to(root).as_posix()
                if relative_mask in referenced_masks:
                    raise ValueError(f"duplicate expert-mask reference: {relative_mask}")
                referenced_masks.add(relative_mask)
                with Image.open(mask_path) as mask:
                    values = {int(value) for value in np.unique(np.asarray(mask))}
                    if mask.mode != "L" or mask.size != image_size:
                        raise ValueError(f"invalid expert mask geometry: {record[key]}")
                    if mask.info or mask.getexif():
                        raise ValueError(f"embedded expert-mask metadata: {record[key]}")
                if not values <= allowed:
                    raise ValueError(f"invalid expert labels: {record[key]}")
    if len(referenced_masks) != expected_tissue * 2:
        raise ValueError("gold-standard manifest does not reference every expert mask once")

    required = (
        "inter_rater/inter_rater_ICC.csv",
        "inter_rater/inter_rater_dice_pairs.csv",
        "inter_rater/inter_rater_proportions.csv",
        "inter_rater/inter_rater_figure1.png",
        "anonymized_votes/form_option_mapping.json",
        "anonymized_votes/google_form_responses.csv",
        "anonymized_votes/selected_doctor_by_image.json",
        "anonymized_votes/vote_ties_report.csv",
        "anonymized_votes/votes_filled.csv",
    )
    for relative in required:
        safe_path(gold_root, relative)


def validate_collections(root: Path) -> None:
    for collection, expected in EXPECTED_COLLECTIONS.items():
        suffixes = {".jpg", ".jpeg"} if collection == "Images" else {".png"}
        actual = count_files(root / collection, suffixes)
        if actual != expected:
            raise ValueError(f"expected {expected} files in {collection}, found {actual}")


def validate_splits(root: Path) -> None:
    expected_counts = {
        "full/labeled.txt": 111,
        "full/unlabeled.txt": 111,
        "1_4/labeled.txt": 27,
        "1_4/unlabeled.txt": 84,
        "1_8/labeled.txt": 13,
        "1_8/unlabeled.txt": 98,
        "1_16/labeled.txt": 6,
        "1_16/unlabeled.txt": 105,
        "validation.txt": 30,
    }
    for relative, expected in expected_counts.items():
        path = root / "splits" / relative
        actual = len([line for line in path.read_text().splitlines() if line.strip()])
        if actual != expected:
            raise ValueError(f"expected {expected} lines in splits/{relative}, found {actual}")
    if (root / "splits" / "fixed").exists():
        raise ValueError("public split directory must use 'full', not 'fixed'")


def validate_text_privacy(root: Path) -> None:
    problems = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE", "VERSION"}:
            continue
        text = path.read_text(errors="replace")
        if ABSOLUTE_PATH.search(text):
            problems.append(f"absolute infrastructure path: {path.relative_to(root)}")
        if EMAIL.search(text):
            problems.append(f"email address: {path.relative_to(root)}")
        if HF_SECRET.search(text):
            problems.append(f"Hugging Face secret: {path.relative_to(root)}")
    if problems:
        raise ValueError("privacy scan failed:\n" + "\n".join(problems))


def validate_checksums(root: Path) -> None:
    checksum_path = root / "checksums.sha256"
    entries = {}
    for line_number, line in enumerate(checksum_path.read_text().splitlines(), start=1):
        fields = line.split("  ", 1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            raise ValueError(f"invalid checksum entry at line {line_number}")
        entries[fields[1]] = fields[0]
    expected_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path != checksum_path
        and path.name != ".gitattributes"
        and ".cache" not in path.relative_to(root).parts
    }
    if set(entries) != expected_paths:
        raise ValueError("checksum manifest does not cover the release exactly")
    for relative, expected in entries.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"checksum mismatch: {relative}")


def validate_release(root: Path) -> None:
    root = root.resolve()
    required = ("README.md", "LICENSE", "VERSION", "class_map.json", "checksums.sha256")
    for filename in required:
        if not (root / filename).is_file():
            raise ValueError(f"missing release document: {filename}")
    symlinks = [path.relative_to(root) for path in root.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"release contains symlinks: {symlinks}")
    validate_collections(root)
    validate_main_records(root)
    validate_gold_standard(root)
    validate_splits(root)
    validate_text_privacy(root)
    validate_checksums(root)
    print("LUTSeg release validation: PASS")
    print("  samples: 141 (111 train, 30 validation; 39 patients)")
    print("  gold standard: 46 cases, 5 clinicians, 460 expert mask files")
    print("  privacy: no EXIF/GPS/XMP, emails, secrets, or absolute local paths")


def main() -> None:
    args = parse_args()
    validate_release(args.release)


if __name__ == "__main__":
    main()
