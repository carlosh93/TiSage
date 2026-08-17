#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from validate_hf_release import validate_release


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLINICIANS = ("user_4", "user_5", "user_6", "user_7", "user_9")
IMAGE_SUFFIXES = {".jpg", ".jpeg"}
PNG_COLLECTIONS = ("Masks", "Masks_RGB", "Wound_Masks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a sanitized, Hugging Face-ready LUTSeg release"
    )
    parser.add_argument("--source", type=Path, required=True, help="complete LUTSeg source")
    parser.add_argument("--output", type=Path, required=True, help="new staging directory")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="TiSage repository containing splits and sanitized voting artifacts",
    )
    return parser.parse_args()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def decoded_digest(path: Path) -> str:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"))
    return sha256_bytes(array.tobytes())


def strip_jpeg_metadata(data: bytes) -> bytes:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("not a JPEG stream")

    output = bytearray(data[:2])
    position = 2
    metadata_markers = {0xE1, 0xED, 0xFE}  # EXIF/XMP, Photoshop, comments
    standalone_markers = {0x01, *range(0xD0, 0xDA)}

    while position < len(data):
        marker_start = position
        if data[position] != 0xFF:
            raise ValueError(f"invalid JPEG marker at byte {position}")
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            raise ValueError("truncated JPEG marker")

        marker = data[position]
        position += 1
        if marker == 0xDA:
            output.extend(data[marker_start:])
            return bytes(output)
        if marker == 0xD9:
            output.extend(data[marker_start:position])
            return bytes(output)
        if marker in standalone_markers:
            output.extend(data[marker_start:position])
            continue
        if position + 2 > len(data):
            raise ValueError("truncated JPEG segment length")

        segment_length = int.from_bytes(data[position : position + 2], "big")
        if segment_length < 2:
            raise ValueError("invalid JPEG segment length")
        segment_end = position + segment_length
        if segment_end > len(data):
            raise ValueError("truncated JPEG segment")
        if marker not in metadata_markers:
            output.extend(data[marker_start:segment_end])
        position = segment_end

    raise ValueError("JPEG stream has no scan data")


def copy_sanitized_jpeg(source: Path, destination: Path) -> None:
    before = decoded_digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(strip_jpeg_metadata(source.read_bytes()))
    after = decoded_digest(destination)
    if before != after:
        raise ValueError(f"decoded pixels changed while sanitizing {source}")


def copy_sanitized_png(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        source_array = np.asarray(image)
        mode = image.mode
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(source_array, mode=mode).save(destination, format="PNG")
    with Image.open(destination) as image:
        output_array = np.asarray(image)
    if source_array.shape != output_array.shape or not np.array_equal(source_array, output_array):
        raise ValueError(f"pixels changed while sanitizing {source}")


def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected image and mask paths")
        pairs.append((fields[0], fields[1]))
    return pairs


def classes_present(mask_path: Path) -> list[int]:
    with Image.open(mask_path) as image:
        values = np.unique(np.asarray(image))
    return [int(value) for value in values]


def copy_main_dataset(source: Path, output: Path) -> list[dict[str, object]]:
    split_pairs = {
        "train": load_pairs(source / "train.txt"),
        "validation": load_pairs(source / "val.txt"),
    }
    records = []
    seen_images = set()
    for split, pairs in split_pairs.items():
        for image_relative, mask_relative in pairs:
            if image_relative in seen_images:
                raise ValueError(f"duplicate image across splits: {image_relative}")
            seen_images.add(image_relative)

            image_source = source / image_relative
            mask_source = source / mask_relative
            if image_source.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"unsupported source image: {image_source}")
            copy_sanitized_jpeg(image_source, output / image_relative)
            copy_sanitized_png(mask_source, output / mask_relative)

            relative_without_suffix = Path(image_relative).relative_to("Images").with_suffix("")
            derived = {}
            for collection in ("Masks_RGB", "Wound_Masks"):
                relative = Path(collection) / relative_without_suffix.with_suffix(".png")
                copy_sanitized_png(source / relative, output / relative)
                derived[collection] = relative.as_posix()

            image_path = Path(image_relative)
            records.append(
                {
                    "image_file_name": image_path.as_posix(),
                    "mask_file_name": Path(mask_relative).as_posix(),
                    "mask_rgb_file_name": derived["Masks_RGB"],
                    "wound_mask_file_name": derived["Wound_Masks"],
                    "split": split,
                    "patient_id": image_path.parent.name,
                    "image_id": image_path.stem,
                    "classes_present": classes_present(mask_source),
                }
            )
    return records


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def copy_split_files(repository_root: Path, output: Path) -> None:
    source_root = repository_root / "splits" / "lutseg"
    for split in ("1_4", "1_8", "1_16", "fixed"):
        public_name = "full" if split == "fixed" else split
        for filename in ("labeled.txt", "unlabeled.txt"):
            source = source_root / split / filename
            destination = output / "splits" / public_name / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source.read_text())
    (output / "splits" / "validation.txt").write_text(
        (source_root / "val.txt").read_text()
    )


def find_expert_mask(root: Path, clinician: str, image_key: str) -> Path:
    image_path = Path(image_key)
    matches = sorted(
        (root / clinician / image_path.parent).glob(f"{image_path.stem}__ann*.png")
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one mask for {clinician}/{image_key}, found {len(matches)}"
        )
    return matches[0]


def copy_gold_standard(source: Path, repository_root: Path, output: Path) -> None:
    form_index = source / "Annotations" / "processed" / "form_review" / "form_index.csv"
    with form_index.open(newline="") as handle:
        gold_rows = list(csv.DictReader(handle))
    if len(gold_rows) != 46:
        raise ValueError(f"expected 46 gold-standard cases, found {len(gold_rows)}")

    tissue_root = source / "Annotations" / "processed" / "masks_by_annotator"
    wound_root = source / "Annotations" / "processed" / "wound_masks_by_annotator"
    manifest = []
    for row in gold_rows:
        image_key = row["image_key"]
        image_path = Path(image_key)
        record: dict[str, object] = {
            "image_file_name": (Path("Images") / image_path).as_posix(),
            "image_key": image_key,
            "patient_id": row["patient_id"],
        }
        for clinician in CLINICIANS:
            tissue_source = find_expert_mask(tissue_root, clinician, image_key)
            wound_source = find_expert_mask(wound_root, clinician, image_key)
            tissue_relative = (
                Path("gold_standard")
                / "masks_by_clinician"
                / clinician
                / "tissue"
                / image_path.with_suffix(".png")
            )
            wound_relative = (
                Path("gold_standard")
                / "masks_by_clinician"
                / clinician
                / "wound"
                / image_path.with_suffix(".png")
            )
            copy_sanitized_png(tissue_source, output / tissue_relative)
            copy_sanitized_png(wound_source, output / wound_relative)
            record[f"{clinician}_tissue_mask"] = tissue_relative.as_posix()
            record[f"{clinician}_wound_mask"] = wound_relative.as_posix()
        manifest.append(record)
    write_jsonl(output / "gold_standard" / "manifest.jsonl", manifest)

    inter_rater_root = output / "gold_standard" / "inter_rater"
    inter_rater_root.mkdir(parents=True, exist_ok=True)
    processed = source / "Annotations" / "processed"
    for filename in (
        "inter_rater_ICC.csv",
        "inter_rater_dice_pairs.csv",
        "inter_rater_proportions.csv",
        "inter_rater_figure1.png",
    ):
        source_path = processed / filename
        destination = inter_rater_root / filename
        if source_path.suffix == ".png":
            copy_sanitized_png(source_path, destination)
        else:
            destination.write_text(source_path.read_text())

    votes_source = repository_root / "LUTSeg" / "examples" / "Form" / "anonymized_votes"
    votes_output = output / "gold_standard" / "anonymized_votes"
    votes_output.mkdir(parents=True, exist_ok=True)
    for source_path in sorted(votes_source.iterdir()):
        if source_path.is_file() and source_path.name != ".gitkeep":
            (votes_output / source_path.name).write_bytes(source_path.read_bytes())


def write_release_documents(repository_root: Path, source: Path, output: Path) -> None:
    card = repository_root / "LUTSeg" / "release" / "HF_DATASET_CARD.md"
    (output / "README.md").write_text(card.read_text())
    (output / "LICENSE").write_text(
        "LUTSeg is licensed under the Creative Commons Attribution 4.0 "
        "International License (CC BY 4.0).\n\n"
        "License text and terms: https://creativecommons.org/licenses/by/4.0/\n"
    )
    class_map = json.loads((source / "class_map.json").read_text())
    (output / "class_map.json").write_text(json.dumps(class_map, indent=2) + "\n")
    (output / "train.txt").write_text((source / "train.txt").read_text())
    (output / "val.txt").write_text((source / "val.txt").read_text())
    (output / "VERSION").write_text("1.0.0\n")


def write_checksums(output: Path) -> None:
    lines = []
    for path in sorted(candidate for candidate in output.rglob("*") if candidate.is_file()):
        if path.name == "checksums.sha256":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output).as_posix()}\n")
    (output / "checksums.sha256").write_text("".join(lines))


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    repository_root = args.repository_root.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"source directory does not exist: {source}")
    if output.exists():
        raise SystemExit(f"output already exists; choose a new staging path: {output}")
    output.mkdir(parents=True)

    records = copy_main_dataset(source, output)
    write_jsonl(output / "metadata.jsonl", records)
    copy_split_files(repository_root, output)
    copy_gold_standard(source, repository_root, output)
    write_release_documents(repository_root, source, output)
    write_checksums(output)
    validate_release(output)
    print(f"Built and validated LUTSeg release: {output}")


if __name__ == "__main__":
    main()
