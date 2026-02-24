#!/usr/bin/env python3
"""
Convert LUTS index masks (0,1..5,255) to color images for visualization.

Output: same folder structure under Masks_RGB/ with one PNG per mask.
Colors: matches luts_visualize.py palette from common.CLASS_COLORS_BGR.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import VALID_MASK_VALUES, CLASS_COLORS_BGR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LUTS masks to RGB for visualization.")
    parser.add_argument(
        "--masks-dir",
        default="data/LUTS/Masks",
        help="Input masks directory (index PNGs).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/LUTS/Masks_RGB",
        help="Output directory for RGB visualizations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    masks_dir = Path(args.masks_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not masks_dir.exists():
        raise SystemExit(f"Masks directory not found: {masks_dir}")

    count = 0
    for mask_path in sorted(masks_dir.rglob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2:
            continue
        unique = set(np.unique(mask).tolist())
        if not unique.issubset(VALID_MASK_VALUES):
            continue

        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        # Use the exact visualization palette as luts_visualize.py.
        for idx, color in CLASS_COLORS_BGR.items():
            colored[mask == idx] = color

        rel = mask_path.relative_to(masks_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), colored)
        count += 1

    print(f"Wrote {count} RGB masks to {output_dir}")


if __name__ == "__main__":
    main()
