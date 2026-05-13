"""Prepare DFUTissue splits for UniMatch-V2.

Default behavior:
  - Labeled pool for ratios: train + val
  - Validation list (val.txt): test set
  - Unlabeled pool: all images in DFUTissue/Unlabeled

Outputs (under splits/dfutissue by default):
  - val.txt: list of "image mask" pairs for evaluation
  - 1_<ratio>/labeled.txt: labeled training pairs for each ratio
  - 1_<ratio>/unlabeled.txt: unlabeled image paths
  - fixed/labeled.txt and fixed/unlabeled.txt (when --fixed-split)

Example:
  # Default behavior (train+val pool, val.txt from test_names.txt):
  python scripts/prepare_dfutissue.py
  python scripts/prepare_dfutissue.py --ratios 64,32,16,8,4
  # Old "clean val" behavior (train-only pool, val.txt from labeled_val_names.txt):
  python scripts/prepare_dfutissue.py --ratio-pool train --val-source val
  # Fixed split (train+val as labeled pool, test as val.txt by default):
  python scripts/prepare_dfutissue.py --fixed-split

  # Historical full generation command:
  python scripts/prepare_dfutissue.py --ratios 32,16,8,4 --fixed-split
"""

import argparse
import random
from pathlib import Path


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def build_pairs(ids: list[str], images_dir: str, masks_dir: str) -> list[str]:
    return [f"{images_dir}/{img_id}.png {masks_dir}/{img_id}.png" for img_id in ids]


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DFUTissue splits for UniMatch-V2")
    parser.add_argument("--data-root", default="data/DFUTissue", help="DFUTissue dataset root")
    parser.add_argument("--splits-dir", default="splits/dfutissue", help="output splits directory")
    parser.add_argument(
        "--ratios",
        default="64,32,16,8",
        help="comma-separated labeled ratios (e.g., 64,32,16,8 for 1/64..1/8)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fixed-split",
        action="store_true",
        help="also generate a fixed split using train+val as labeled pool",
    )
    parser.add_argument(
        "--ratio-pool",
        choices=["train", "trainval"],
        default="trainval",
        help="labeled pool for ratio splits (train or train+val)",
    )
    parser.add_argument(
        "--val-source",
        choices=["val", "test"],
        default="test",
        help="use labeled_val_names.txt or test_names.txt for val.txt",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    labeled_root = data_root / "Labeled"
    splits_dir = Path(args.splits_dir)

    train_ids = read_ids(labeled_root / "labeled_train_names.txt")
    val_ids = read_ids(labeled_root / "labeled_val_names.txt")
    test_ids = read_ids(labeled_root / "test_names.txt")

    train_images_dir = "Labeled/Original/Images/TrainVal"
    train_masks_dir = "Labeled/Original/Annotations/TrainVal"
    test_images_dir = "Labeled/Original/Images/Test"
    test_masks_dir = "Labeled/Original/Annotations/Test"

    val_source_ids = val_ids if args.val_source == "val" else test_ids
    val_pairs = build_pairs(val_source_ids, train_images_dir, train_masks_dir)
    if args.val_source == "test":
        val_pairs = build_pairs(val_source_ids, test_images_dir, test_masks_dir)

    write_lines(splits_dir / "val.txt", val_pairs)

    # Unlabeled images (no masks needed in train_u)
    unlabeled_paths = sorted((data_root / "Unlabeled").glob("*.png"))
    unlabeled_rel = [f"Unlabeled/{p.name}" for p in unlabeled_paths]

    ratios = [int(r) for r in args.ratios.split(",") if r.strip()]
    rng = random.Random(args.seed)
    ratio_pool_ids = train_ids if args.ratio_pool == "train" else train_ids + val_ids
    if args.ratio_pool == "trainval" and args.val_source == "val":
        print("Warning: ratio-pool=trainval reuses val IDs in training.")

    for ratio in ratios:
        k = max(1, len(ratio_pool_ids) // ratio)
        labeled_sample = rng.sample(ratio_pool_ids, k)
        labeled_pairs = build_pairs(labeled_sample, train_images_dir, train_masks_dir)

        split_dir = splits_dir / f"1_{ratio}"
        write_lines(split_dir / "labeled.txt", labeled_pairs)
        write_lines(split_dir / "unlabeled.txt", unlabeled_rel)

    if args.fixed_split:
        fixed_ids = train_ids + val_ids
        fixed_pairs = build_pairs(fixed_ids, train_images_dir, train_masks_dir)
        fixed_dir = splits_dir / "fixed"
        write_lines(fixed_dir / "labeled.txt", fixed_pairs)
        write_lines(fixed_dir / "unlabeled.txt", unlabeled_rel)

    print(f"Prepared splits in: {splits_dir}")


if __name__ == "__main__":
    main()
