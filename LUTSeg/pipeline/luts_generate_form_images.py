#!/usr/bin/env python3
"""
Generate anonymized review-form images for doctor voting.

Layout:
- Fixed 2x3 grid (2 rows, 3 columns = 6 panels).
- Panel 1: Original image (no mask).
- Panels 2-6: Option A/B/C/D/E, one per annotator (overlay on original image).
- With 4 doctors: 5 panels filled (Original + A,B,C,D), one blank.
- With 5 doctors: all 6 panels filled (Original + A,B,C,D,E).

Outputs:
- <output-dir>/images/img_XXXX.png (one composite per image)
- <output-dir>/form_option_mapping.json (image_id + option -> doctor_id mapping)
- <output-dir>/form_index.csv
- <output-dir>/votes_template.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from common import CLASS_COLORS_BGR, CLASS_ID_TO_NAME, parse_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate anonymized form images for label voting.")
    parser.add_argument(
        "--groups-json",
        default="data/LUTS/Annotations/processed/image_groups.json",
        help="Image groups JSON path (step 3 output).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/LUTS/Annotations/processed/form_review",
        help="Output directory for images + mapping files.",
    )
    parser.add_argument(
        "--golden-only",
        action="store_true",
        default=True,
        help="Use only golden-set groups (default: True).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for option shuffling (anonymization order).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Overlay alpha for tissue colors on original image.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=560,
        help="Panel width in composite grid.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=420,
        help="Panel height in composite grid.",
    )
    return parser.parse_args()


def _safe_read_image(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def _safe_read_mask(path: str) -> np.ndarray | None:
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None or m.ndim != 2:
        return None
    return m.astype(np.uint8)


def _fit_pad(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if ih <= 0 or iw <= 0:
        return np.full((h, w, 3), 240, dtype=np.uint8)
    scale = min(w / iw, h / ih)
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), 245, dtype=np.uint8)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    ih, iw = image_bgr.shape[:2]
    if mask.shape[:2] != (ih, iw):
        mask = cv2.resize(mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

    color = np.zeros_like(image_bgr)
    for cid in (1, 2, 3, 4, 5):
        c = CLASS_COLORS_BGR[cid]
        color[mask == cid] = c

    out = image_bgr.copy()
    tissue = (mask >= 1) & (mask <= 5)
    if np.any(tissue):
        out_f = out.astype(np.float32)
        color_f = color.astype(np.float32)
        out_f[tissue] = (1.0 - alpha) * out_f[tissue] + alpha * color_f[tissue]
        out = np.clip(out_f, 0, 255).astype(np.uint8)
    return out


def _make_legend_strip(max_width: int) -> np.ndarray:
    legend_ids = (1, 2, 3, 4, 5)
    box_w = 30
    box_h = 20
    pad = 10
    text_gap = 8
    item_gap = 18
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 1

    specs: list[tuple[int, str, int, int]] = []
    total_w = pad
    for cid in legend_ids:
        name = CLASS_ID_TO_NAME.get(cid, f"Class {cid}")
        (tw, th), _ = cv2.getTextSize(name, font, font_scale, text_thickness)
        item_w = box_w + text_gap + tw + item_gap
        specs.append((cid, name, th, item_w))
        total_w += item_w
    total_w += pad
    h = box_h + 2 * pad

    legend = np.full((h, total_w, 3), 245, dtype=np.uint8)
    x = pad
    mid_y = pad + box_h // 2
    for cid, name, text_h, item_w in specs:
        cv2.rectangle(legend, (x, pad), (x + box_w, pad + box_h), CLASS_COLORS_BGR[cid], -1)
        cv2.rectangle(legend, (x, pad), (x + box_w, pad + box_h), (120, 120, 120), 1)
        cv2.putText(
            legend,
            name,
            (x + box_w + text_gap, mid_y + text_h // 2),
            font,
            font_scale,
            (20, 20, 20),
            text_thickness,
            cv2.LINE_AA,
        )
        x += item_w

    if total_w <= max_width:
        return legend

    scale = max_width / float(total_w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(legend, (max_width, new_h), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=16)
def _legend_strip_for_width(max_width: int) -> np.ndarray:
    return _make_legend_strip(max_width=max_width)


def _draw_grid_2x3(panels: list[tuple[str, np.ndarray]], tile_w: int, tile_h: int) -> np.ndarray:
    """2 rows x 3 columns = 6 panels (original + up to 5 options A–E)."""
    cols = 3
    rows = 2
    pad = 20
    header_h = 44
    legend = _legend_strip_for_width(max(200, cols * tile_w + (cols - 1) * pad))

    full_h = rows * (tile_h + header_h) + (rows + 1) * pad + legend.shape[0] + pad
    full_w = cols * tile_w + (cols + 1) * pad
    canvas = np.full((full_h, full_w, 3), 235, dtype=np.uint8)

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (tile_w + pad)
        y0 = pad + r * (tile_h + header_h + pad)
        x1 = x0 + tile_w
        yh = y0 + header_h
        y1 = yh + tile_h

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (180, 180, 180), 1)
        cv2.rectangle(canvas, (x0, y0), (x1, yh), (220, 220, 220), -1)

        if idx < len(panels):
            title, panel_img = panels[idx]
            cv2.putText(
                canvas,
                title,
                (x0 + 12, y0 + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            fitted = _fit_pad(panel_img, tile_w, tile_h)
            canvas[yh:y1, x0:x1] = fitted
        else:
            cv2.putText(
                canvas,
                "N/A",
                (x0 + 12, y0 + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (120, 120, 120),
                2,
                cv2.LINE_AA,
            )
            canvas[yh:y1, x0:x1] = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)

    ly0 = rows * (tile_h + header_h) + (rows + 1) * pad
    lx0 = (full_w - legend.shape[1]) // 2
    canvas[ly0 : ly0 + legend.shape[0], lx0 : lx0 + legend.shape[1]] = legend

    return canvas


def main() -> None:
    args = parse_args()
    groups = parse_json(args.groups_json)
    if not isinstance(groups, list) or not groups:
        raise SystemExit("Image groups JSON is empty or invalid.")

    if args.golden_only:
        groups = [g for g in groups if bool(g.get("is_golden_patient"))]
    if not groups:
        raise SystemExit("No groups selected for form generation.")

    rng = random.Random(args.seed)
    out_root = Path(args.output_dir).resolve()
    img_dir = out_root / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    mapping_items: list[dict] = []
    index_rows: list[dict] = []
    vote_rows: list[dict] = []

    option_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    image_counter = 1

    for group in groups:
        image_key = str(group.get("image_key") or "")
        patient_id = str(group.get("patient_id") or "")
        source_path = group.get("source_image_path")
        anns = list(group.get("annotations", []) or [])
        if not image_key or not source_path or not anns:
            continue

        src = _safe_read_image(source_path)
        if src is None:
            continue

        anns = sorted(anns, key=lambda a: str(a.get("doctor_id") or ""))
        rng.shuffle(anns)

        panels: list[tuple[str, np.ndarray]] = [("Original", src)]
        options: dict[str, dict] = {}
        labels_for_image: list[str] = []

        for idx, ann in enumerate(anns):
            if idx >= len(option_labels):
                break
            label = option_labels[idx]
            labels_for_image.append(label)
            doctor_id = str(ann.get("doctor_id") or "")
            mask_path = ann.get("mask_path")
            if not doctor_id or not mask_path:
                continue
            mask = _safe_read_mask(mask_path)
            if mask is None:
                continue
            overlay = _overlay_mask(src, mask, alpha=float(np.clip(args.alpha, 0.0, 1.0)))
            panels.append((f"Option {label}", overlay))
            options[label] = {
                "doctor_id": doctor_id,
                "annotation_id": ann.get("annotation_id"),
                "task_id": ann.get("task_id"),
                "project_id": ann.get("project_id"),
                "mask_path": mask_path,
                "wound_mask_path": ann.get("wound_mask_path"),
            }

        if not options:
            continue

        composite = _draw_grid_2x3(panels, tile_w=args.tile_width, tile_h=args.tile_height)
        image_id = f"img_{image_counter:04d}"
        image_counter += 1
        out_path = img_dir / f"{image_id}.png"
        cv2.imwrite(str(out_path), composite)

        mapping_items.append(
            {
                "image_id": image_id,
                "image_key": image_key,
                "patient_id": patient_id,
                "form_image": str(out_path),
                "options": options,
            }
        )
        index_rows.append(
            {
                "image_id": image_id,
                "image_key": image_key,
                "patient_id": patient_id,
                "form_image": str(out_path),
                "option_labels": ",".join(sorted(options.keys())),
            }
        )
        vote_rows.append(
            {
                "image_id": image_id,
                "image_key": image_key,
                "selected_option": "",
            }
        )

    if not mapping_items:
        raise SystemExit("No form images were generated.")

    mapping_json = out_root / "form_option_mapping.json"
    write_json(
        mapping_json,
        {
            "meta": {
                "groups_json": args.groups_json,
                "golden_only": args.golden_only,
                "seed": args.seed,
                "alpha": args.alpha,
                "tile_width": args.tile_width,
                "tile_height": args.tile_height,
            },
            "items": mapping_items,
        },
    )

    index_csv = out_root / "form_index.csv"
    with open(index_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image_id", "image_key", "patient_id", "form_image", "option_labels"]
        )
        writer.writeheader()
        writer.writerows(index_rows)

    votes_csv = out_root / "votes_template.csv"
    with open(votes_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_key", "selected_option"])
        writer.writeheader()
        writer.writerows(vote_rows)

    print(f"Generated form images: {len(mapping_items)}")
    print(f"Images dir: {img_dir}")
    print(f"Option mapping: {mapping_json}")
    print(f"Index CSV: {index_csv}")
    print(f"Votes template: {votes_csv}")


if __name__ == "__main__":
    main()
