#!/usr/bin/env python3
"""
Step 6: Export final LUTS dataset layout.

Creates:
- data/LUTS/Images/**
- data/LUTS/Masks/**
- data/LUTS/Wound_Masks/**
- data/LUTS/train.txt
- data/LUTS/val.txt
- data/LUTS/class_map.json
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import parse_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LUTS dataset layout.")
    parser.add_argument(
        "--splits-json",
        default="data/LUTS/Annotations/processed/splits.json",
        help="Input splits JSON path.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/LUTS",
        help="Output dataset root.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How to place images and masks into dataset layout.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split txt files.",
    )
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    if src_resolved == dst_resolved:
        return  # already in place (e.g. consensus masks already in data/LUTS/Masks)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "hardlink":
        dst.hardlink_to(src.resolve())
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    args = parse_args()
    splits = parse_json(args.splits_json)
    if not isinstance(splits, dict):
        raise SystemExit("Invalid splits JSON.")

    dataset_root = Path(args.dataset_root).resolve()
    images_root = dataset_root / "Images"
    masks_root = dataset_root / "Masks"
    wound_masks_root = dataset_root / "Wound_Masks"
    images_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)
    wound_masks_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        txt_path = dataset_root / f"{split}.txt"
        if txt_path.exists() and not args.overwrite:
            raise SystemExit(
                f"{txt_path} exists. Use --overwrite or remove existing split files."
            )

    split_lines = {"train": [], "val": []}
    samples_by_split = splits.get("samples", {})

    for split in ("train", "val"):
        for sample in samples_by_split.get(split, []):
            patient_id = sample.get("patient_id") or "unknown_patient"
            image_name = sample.get("image_name")
            src_image = sample.get("source_image_path")
            src_mask = sample.get("mask_path")
            src_wound_mask = sample.get("wound_mask_path")
            if not image_name or not src_image or not src_mask:
                continue

            src_image_path = Path(src_image)
            src_mask_path = Path(src_mask)
            if not src_image_path.exists() or not src_mask_path.exists():
                continue

            dst_image = images_root / patient_id / image_name
            dst_mask = masks_root / patient_id / f"{Path(image_name).stem}.png"
            dst_wound_mask = wound_masks_root / patient_id / f"{Path(image_name).stem}.png"

            link_or_copy(src_image_path, dst_image, args.link_mode)
            link_or_copy(src_mask_path, dst_mask, args.link_mode)
            if src_wound_mask:
                src_wound_path = Path(src_wound_mask)
                if src_wound_path.exists():
                    link_or_copy(src_wound_path, dst_wound_mask, args.link_mode)

            rel_image = dst_image.relative_to(dataset_root).as_posix()
            rel_mask = dst_mask.relative_to(dataset_root).as_posix()
            split_lines[split].append(f"{rel_image} {rel_mask}")

    for split in ("train", "val"):
        txt_path = dataset_root / f"{split}.txt"
        txt_path.write_text("\n".join(split_lines[split]) + "\n", encoding="utf-8")

    class_map = {
        "background": 0,
        "Epithelial tissue": 1,
        "Slough": 2,
        "Granulation tissue": 3,
        "Necrotic tissue": 4,
        "Other": 5,
        "ignore": 255,
    }
    write_json(dataset_root / "class_map.json", class_map)

    print(f"Dataset exported to: {dataset_root}")
    print("Split sizes:", {split: len(split_lines[split]) for split in ("train", "val")})


if __name__ == "__main__":
    main()
