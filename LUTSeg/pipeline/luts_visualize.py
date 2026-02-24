#!/usr/bin/env python3
"""
Visualize LUTS images and masks (overlay or side-by-side) to spot-check correctness.

Usage:
  python data/LUTS/pipeline/luts_visualize.py
  python data/LUTS/pipeline/luts_visualize.py --split val --limit 20
  python data/LUTS/pipeline/luts_visualize.py --save-dir data/LUTS/check_samples --limit 30

Keys: n/p = next, b = previous, q = quit.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# Optional matplotlib for interactive window
try:
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    mpatches = None

from common import GOLDEN_PATIENTS, IGNORE_VALUE, CLASS_ID_TO_NAME, CLASS_COLORS_BGR

# For legend: id -> display name
CLASS_ID_TO_LEGEND_NAME = {
    0: "Background",
    **{k: v for k, v in CLASS_ID_TO_NAME.items()},
    IGNORE_VALUE: "Ignore",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize LUTS image + mask.")
    parser.add_argument(
        "--dataset-root",
        default="data/LUTS",
        help="LUTS dataset root (contains train.txt, Images/, Masks/).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Which split to visualize.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of samples to load (0 = all).",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="If set, save overlay previews here and exit (no interactive window).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Overlay opacity (0=transparent mask, 1=full mask color). Default 0.65 for better contrast.",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Do not show or save the color legend (tissue labels at bottom).",
    )
    parser.add_argument(
        "--include-golden",
        action="store_true",
        help="Include golden-set patients in visualization (by default they are excluded).",
    )
    return parser.parse_args()


def mask_to_overlay_bgr(mask: np.ndarray) -> np.ndarray:
    """Build BGR overlay from index mask."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, bgr in CLASS_COLORS_BGR.items():
        out[mask == idx] = bgr
    return out


def blend_overlay_only_on_labels(
    img: np.ndarray, overlay: np.ndarray, mask: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Blend overlay onto image only where mask has a label (1..5 or 255).
    Background (0) stays the original image so the rest of the image is not darkened.
    """
    out = img.copy().astype(np.float32)
    overlay_f = overlay.astype(np.float32)
    img_f = img.astype(np.float32)
    # Only blend on labeled pixels (non-background: 1..5 and 255)
    labeled = (mask > 0)
    labeled = labeled[:, :, np.newaxis]
    blended = alpha * overlay_f + (1.0 - alpha) * img_f
    out = np.where(labeled, blended, img_f)
    return np.clip(out, 0, 255).astype(np.uint8)


def make_legend_image_bgr(box_h: int = 28, pad: int = 8, font_scale: float = 0.5) -> np.ndarray:
    """Build a horizontal legend strip (BGR) to paste below previews."""
    order = [0, 1, 2, 3, 4, 5, IGNORE_VALUE]
    box_w = 32
    gap = 4
    # Approximate character width for FONT_HERSHEY_SIMPLEX at font_scale 0.5
    char_w = 7
    total_w = pad * 2
    for idx in order:
        name = CLASS_ID_TO_LEGEND_NAME.get(idx, str(idx))
        total_w += box_w + gap + max(len(name) * char_w, 40)
    w = total_w
    h = box_h + pad * 2
    legend = np.ones((h, w, 3), dtype=np.uint8) * 255
    x = pad
    for idx in order:
        color = CLASS_COLORS_BGR[idx]
        name = CLASS_ID_TO_LEGEND_NAME.get(idx, str(idx))
        cv2.rectangle(legend, (x, pad), (x + box_w, pad + box_h), color, -1)
        cv2.rectangle(legend, (x, pad), (x + box_w, pad + box_h), (128, 128, 128), 1)
        cv2.putText(
            legend, name, (x + box_w + 2, pad + box_h - 6),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
        )
        x += box_w + gap + max(len(name) * char_w, 40)
    return legend


def load_lines(
    dataset_root: Path, split: str, limit: int, include_golden: bool = False
) -> list[tuple[str, str]]:
    txt = dataset_root / f"{split}.txt"
    if not txt.exists():
        return []
    lines = txt.read_text(encoding="utf-8").strip().splitlines()
    pairs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        rel_image = parts[0]
        if not include_golden:
            # rel_image is e.g. Images/Patient_1/P1_T1.jpeg
            path_parts = rel_image.replace("\\", "/").split("/")
            if len(path_parts) >= 2 and path_parts[1] in GOLDEN_PATIENTS:
                continue
        pairs.append((parts[0], parts[1]))
    if limit > 0:
        pairs = pairs[:limit]
    return pairs


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")

    pairs = load_lines(root, args.split, args.limit, include_golden=args.include_golden)
    if not pairs:
        raise SystemExit(f"No samples in {args.split}.txt (or limit=0).")

    alpha = float(np.clip(args.alpha, 0.0, 1.0))

    if args.save_dir:
        save_dir = Path(args.save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        # Always build legend so we can save legend.png (reference) even with --no-legend.
        legend_strip = make_legend_image_bgr()
        lh, lw = legend_strip.shape[:2]
        saved = 0
        for i, (rel_image, rel_mask) in enumerate(pairs):
            img_path = root / rel_image
            mask_path = root / rel_mask
            if not img_path.exists() or not mask_path.exists():
                continue
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if img is None or mask is None:
                continue
            overlay = mask_to_overlay_bgr(mask)
            blend = blend_overlay_only_on_labels(img, overlay, mask, alpha)
            if not args.no_legend:
                if blend.shape[1] != lw:
                    scale = blend.shape[1] / max(lw, 1)
                    new_lw = int(lw * scale)
                    new_lh = int(lh * scale)
                    leg_resized = cv2.resize(legend_strip, (new_lw, new_lh))
                else:
                    leg_resized = legend_strip
                h, w = blend.shape[:2]
                lh2, lw2 = leg_resized.shape[:2]
                out_h = h + lh2
                out_w = max(w, lw2)
                out_img = np.ones((out_h, out_w, 3), dtype=np.uint8) * 255
                out_img[:h, :w] = blend
                out_img[h : h + lh2, :lw2] = leg_resized
            else:
                out_img = blend
            name = Path(rel_image).stem
            out_path = save_dir / f"{i+1:04d}_{name}.png"
            cv2.imwrite(str(out_path), out_img)
            saved += 1
        # Always write legend.png as reference (even with --no-legend).
        cv2.imwrite(str(save_dir / "legend.png"), legend_strip)
        print(f"Saved {saved} previews + legend.png to {save_dir}")
        return

    if not HAS_MPL:
        print("Install matplotlib for interactive view, or use --save-dir to export previews.")
        return

    index = [0]

    def refresh():
        idx = index[0] % len(pairs)
        rel_image, rel_mask = pairs[idx]
        img_path = root / rel_image
        mask_path = root / rel_mask
        if not img_path.exists() or not mask_path.exists():
            return None, None, None
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None or mask is None:
            return None, None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = mask_to_overlay_bgr(mask)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        blend = blend_overlay_only_on_labels(img, overlay, mask, alpha)
        blend_rgb = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)
        return img_rgb, overlay_rgb, blend_rgb

    def on_key(event):
        if event.key in ("n", "right", " "):
            index[0] += 1
        elif event.key in ("b", "left"):
            index[0] -= 1
        elif event.key == "q":
            plt.close(fig)
            return
        else:
            return
        idx = index[0] % len(pairs)
        rel_image, rel_mask = pairs[idx]
        img_rgb, ov_rgb, blend_rgb = refresh()
        if img_rgb is None:
            return
        ax0.imshow(img_rgb)
        ax1.imshow(ov_rgb)
        ax2.imshow(blend_rgb)
        fig.suptitle(f"{index[0] + 1}/{len(pairs)}  {rel_image}")
        fig.canvas.draw_idle()

    img_rgb, ov_rgb, blend_rgb = refresh()
    if img_rgb is None:
        raise SystemExit("Could not load first sample.")
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 5))
    ax0.set_title("Image")
    ax0.imshow(img_rgb)
    ax1.set_title("Mask (RGB)")
    ax1.imshow(ov_rgb)
    if not args.no_legend:
        legend_order = [0, 1, 2, 3, 4, 5, IGNORE_VALUE]
        patches = []
        for idx in legend_order:
            b, g, r = CLASS_COLORS_BGR[idx]
            patches.append(mpatches.Patch(color=(r / 255, g / 255, b / 255), label=CLASS_ID_TO_LEGEND_NAME.get(idx, str(idx))))
        ax1.legend(handles=patches, loc="upper left", fontsize=7)
    ax2.set_title("Overlay")
    ax2.imshow(blend_rgb)
    fig.suptitle(f"1/{len(pairs)}  {pairs[0][0]}")
    plt.tight_layout()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
