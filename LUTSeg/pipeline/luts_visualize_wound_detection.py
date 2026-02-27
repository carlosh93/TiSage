#!/usr/bin/env python3
"""
Visualize wound detection labels (from wound_outline) with contour + bounding box.

Reads split entries from train.txt/val.txt, then uses:
- image path from split
- wound mask path inferred as Masks/... -> Wound_Masks/...

Output:
- one visualization image per sample in --save-dir
- optional collage grid image with --grid-path
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from common import GOLDEN_PATIENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize wound outline + bounding box from Wound_Masks."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/LUTS",
        help="Dataset root containing train.txt/val.txt, Images/, Masks/, Wound_Masks/.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="val",
        help="Split to visualize.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max samples to visualize (0 = all).",
    )
    parser.add_argument(
        "--save-dir",
        default="luts_wound_detection_previews",
        help="Directory to save per-image visualizations.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Overlay alpha for wound fill (0=transparent, 1=solid). Only the wound region is blended; rest of image unchanged. Default 0.35.",
    )
    parser.add_argument(
        "--grid-path",
        default="",
        help="Optional output path for a collage grid image.",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=6,
        help="Number of columns in optional collage grid.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=240,
        help="Tile width for optional collage grid.",
    )
    parser.add_argument(
        "--include-golden",
        action="store_true",
        help="Include golden-set patients in visualization (by default they are excluded).",
    )
    return parser.parse_args()


def load_pairs(
    dataset_root: Path, split: str, limit: int, include_golden: bool = False
) -> list[tuple[str, str]]:
    txt_path = dataset_root / f"{split}.txt"
    if not txt_path.exists():
        return []
    pairs: list[tuple[str, str]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        rel_image = parts[0]
        if not include_golden:
            path_parts = rel_image.replace("\\", "/").split("/")
            if len(path_parts) >= 2 and path_parts[1] in GOLDEN_PATIENTS:
                continue
        pairs.append((parts[0], parts[1]))
    if limit > 0:
        pairs = pairs[:limit]
    return pairs


def derive_wound_relpath(mask_rel: str) -> str:
    p = Path(mask_rel).as_posix()
    return p.replace("Masks/", "Wound_Masks/", 1)


def draw_wound_detection(
    image_bgr: np.ndarray,
    wound_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    out = image_bgr.copy().astype(np.float32)
    wound_bin = (wound_mask > 0).astype(np.uint8)
    if wound_bin.max() == 0:
        return image_bgr.copy()

    # Overlay only on wound region: blue fill to match outline/bbox (BGR).
    fill_color = np.array([255, 0, 0], dtype=np.float32)  # BGR blue
    wound_3 = wound_bin[:, :, np.newaxis].astype(np.float32)
    img_f = image_bgr.astype(np.float32)
    blended = alpha * fill_color + (1.0 - alpha) * img_f
    out = np.where(wound_3 > 0, blended, img_f)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Blue for outline and bbox (BGR). Scale thickness with image size for consistent look.
    outline_color = (255, 0, 0)
    h, w = out.shape[:2]
    thickness = max(1, min(8, round(min(h, w) / 400)))
    contours, _ = cv2.findContours(wound_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, outline_color, thickness)
        merged = np.vstack(contours)
        x, y, w_rect, h_rect = cv2.boundingRect(merged)
        cv2.rectangle(out, (x, y), (x + w_rect, y + h_rect), outline_color, thickness)
    return out


def save_collage(image_paths: list[Path], out_path: Path, cols: int, tile_width: int) -> None:
    if not image_paths:
        return
    imgs = []
    tile_height = None
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        new_h = int(round((tile_width / max(w, 1)) * h))
        resized = cv2.resize(img, (tile_width, new_h), interpolation=cv2.INTER_AREA)
        imgs.append(resized)
        if tile_height is None:
            tile_height = new_h
        else:
            tile_height = min(tile_height, new_h)
    if not imgs or tile_height is None:
        return

    norm = [cv2.resize(img, (tile_width, tile_height), interpolation=cv2.INTER_AREA) for img in imgs]
    cols = max(1, cols)
    rows = int(math.ceil(len(norm) / cols))
    canvas = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)
    for i, img in enumerate(norm):
        r = i // cols
        c = i % cols
        y0 = r * tile_height
        x0 = c * tile_width
        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = img

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(dataset_root, args.split, args.limit, include_golden=args.include_golden)
    if not pairs:
        raise SystemExit(f"No samples found in {dataset_root / (args.split + '.txt')}")

    written: list[Path] = []
    for i, (rel_image, rel_mask) in enumerate(pairs, start=1):
        img_path = dataset_root / rel_image
        wound_path = dataset_root / derive_wound_relpath(rel_mask)
        if not img_path.exists() or not wound_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        wound = cv2.imread(str(wound_path), cv2.IMREAD_UNCHANGED)
        if img is None or wound is None or wound.ndim != 2:
            continue

        vis = draw_wound_detection(img, wound, alpha=float(np.clip(args.alpha, 0.0, 1.0)))
        out_name = f"{i:04d}_{Path(rel_image).stem}.png"
        out_path = save_dir / out_name
        cv2.imwrite(str(out_path), vis)
        written.append(out_path)

    if args.grid_path:
        save_collage(written, Path(args.grid_path).resolve(), args.grid_cols, args.tile_width)
        print(f"Saved collage: {Path(args.grid_path).resolve()}")
    print(f"Saved {len(written)} wound detection previews to {save_dir}")


if __name__ == "__main__":
    main()
