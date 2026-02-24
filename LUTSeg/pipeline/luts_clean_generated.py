#!/usr/bin/env python3
"""
Remove all pipeline-generated files so you can run the full pipeline from scratch.

Does NOT delete:
- data/LUTS/Annotations/raw/ (your Label Studio exports)

Deletes:
- data/LUTS/Annotations/processed/
- data/LUTS/Masks/
- data/LUTS/Masks_RGB/
- data/LUTS/Wound_Masks/
- data/LUTS/Images/
- data/LUTS/train.txt, val.txt (and legacy test.txt if present)
- data/LUTS/class_map.json
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete all LUTS pipeline-generated files (keeps Annotations/raw)."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/LUTS",
        help="LUTS dataset root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be removed.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    to_remove: list[Path] = []
    for name in (
        "Annotations/processed",
        "Masks",
        "Masks_RGB",
        "Wound_Masks",
        "Images",
    ):
        p = root / name
        if p.exists():
            to_remove.append(p)
    for name in ("train.txt", "val.txt", "test.txt", "class_map.json"):
        p = root / name
        if p.exists():
            to_remove.append(p)

    if not to_remove:
        print("Nothing to remove.")
        return

    print("Would remove:")
    for p in to_remove:
        print(f"  {p}")
    if args.dry_run:
        return
    if not args.yes:
        reply = input("Proceed? [y/N]: ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return

    for p in to_remove:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        print(f"Removed: {p}")
    print("Done. You can run the full pipeline again.")


if __name__ == "__main__":
    main()
