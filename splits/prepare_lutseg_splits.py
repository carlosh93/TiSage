#!/usr/bin/env python3
"""
Prepare UniMatch-V2 split files for LUTSeg.

Input layout (under data/LUTSeg):
  - train.txt  : "<image_rel_path> <mask_rel_path>"
  - val.txt    : "<image_rel_path> <mask_rel_path>"

Output layout (under splits/lutseg by default):
  - val.txt
  - 1_<ratio>/labeled.txt
  - 1_<ratio>/unlabeled.txt
  - fixed/labeled.txt and fixed/unlabeled.txt (optional)
"""

import argparse
import random
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def image_rel_from_pair(pair_line: str) -> str:
    parts = pair_line.split()
    if len(parts) < 1:
        raise ValueError(f"Invalid split line: '{pair_line}'")
    return parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LUTSeg splits for UniMatch-V2")
    parser.add_argument("--data-root", default="data/LUTSeg", help="LUTSeg dataset root")
    parser.add_argument("--output-dir", default="splits/lutseg", help="Output split directory")
    parser.add_argument(
        "--ratios",
        default="4,8,16",
        help="Comma-separated labeled ratios: 1/<ratio> (e.g. 4,8,16)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument(
        "--fixed-split",
        action="store_true",
        help="Also write fixed/ split where all train pairs are labeled.",
    )
    parser.add_argument(
        "--use-full-unlabeled",
        action="store_true",
        help="Use all train images as unlabeled pool instead of only the non-labeled remainder.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    train_path = data_root / "train.txt"
    val_path = data_root / "val.txt"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected train/val files at '{train_path}' and '{val_path}'."
        )

    train_pairs = read_lines(train_path)
    val_pairs = read_lines(val_path)
    if len(train_pairs) == 0 or len(val_pairs) == 0:
        raise ValueError("train.txt/val.txt must be non-empty.")

    # Validation is copied as-is for eval loader (splits/<dataset>/val.txt).
    write_lines(output_dir / "val.txt", val_pairs)

    ratios = [int(r.strip()) for r in args.ratios.split(",") if r.strip()]
    if not ratios:
        raise ValueError("At least one ratio must be provided.")

    rng = random.Random(args.seed)
    all_indices = list(range(len(train_pairs)))

    for ratio in ratios:
        if ratio <= 0:
            raise ValueError(f"Ratio denominator must be > 0, got {ratio}")
        n_labeled = max(1, len(train_pairs) // ratio)
        labeled_idx = sorted(rng.sample(all_indices, n_labeled))
        labeled_set = set(labeled_idx)

        labeled_pairs = [train_pairs[i] for i in labeled_idx]
        if args.use_full_unlabeled:
            unlabeled_images = [image_rel_from_pair(x) for x in train_pairs]
        else:
            unlabeled_images = [
                image_rel_from_pair(train_pairs[i])
                for i in all_indices
                if i not in labeled_set
            ]

        split_dir = output_dir / f"1_{ratio}"
        write_lines(split_dir / "labeled.txt", labeled_pairs)
        write_lines(split_dir / "unlabeled.txt", unlabeled_images)

        print(
            f"[1_{ratio}] labeled={len(labeled_pairs)} "
            f"unlabeled={len(unlabeled_images)} total_train={len(train_pairs)}"
        )

    if args.fixed_split:
        fixed_dir = output_dir / "fixed"
        fixed_labeled = train_pairs
        fixed_unlabeled = [image_rel_from_pair(x) for x in train_pairs]
        write_lines(fixed_dir / "labeled.txt", fixed_labeled)
        write_lines(fixed_dir / "unlabeled.txt", fixed_unlabeled)
        print(
            f"[fixed] labeled={len(fixed_labeled)} unlabeled={len(fixed_unlabeled)} "
            f"total_train={len(train_pairs)}"
        )

    print(f"Wrote LUTSeg splits to: {output_dir}")


if __name__ == "__main__":
    main()
