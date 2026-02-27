#!/usr/bin/env python3
"""
Evaluate MedSigLIP zero-shot superpixel predictions on LUTSeg validation set.

This mirrors the superpixel pipeline used in train_prior_lutseg.py
so results are directly comparable against the trained classifier head.

Usage:
  python3 eval_prior_zeroshot_lutseg.py
  python3 eval_prior_zeroshot_lutseg.py --n-segments 128 --min-size 50
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

from train_prior_lutseg import (
    CLASS_NAMES,
    NUM_CLASSES,
    CROP_SIZE,
    HAS_SKIMAGE,
    compute_iou_per_class,
    extract_superpixel_crops,
    load_gt_mask,
    segment_superpixels,
)


TISSUE_TEXTS = [
    "a clinical photo of healthy skin without a wound",
    "a clinical photo of epithelial tissue in a wound",
    "a clinical photo of slough tissue in a wound",
    "a clinical photo of granulation tissue in a wound",
    "a clinical photo of necrotic tissue in a wound",
    "a clinical photo of other tissue in a wound",
]


@torch.no_grad()
def run_medsiglip_superpixels_zeroshot(
    model,
    processor,
    img_pil,
    device,
    n_segments=64,
    compactness=10.0,
    min_size=100,
    context_margin=2,
):
    arr = np.array(img_pil)
    labels_sp = segment_superpixels(arr, n_segments=n_segments, compactness=compactness)
    crops, kept_ids = extract_superpixel_crops(
        arr,
        labels_sp,
        min_size=min_size,
        target_size=CROP_SIZE,
        preserve_aspect=True,
        context_margin=context_margin,
    )
    if not crops:
        return None, labels_sp

    inputs = processor(
        text=TISSUE_TEXTS,
        images=crops,
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits_per_image, dim=1).cpu().numpy()  # (N, C)

    pred_map = np.zeros_like(labels_sp, dtype=np.int32)
    id_to_idx = {lid: i for i, lid in enumerate(kept_ids)}
    for lid in np.unique(labels_sp):
        if lid in id_to_idx:
            pred_map[labels_sp == lid] = probs[id_to_idx[lid]].argmax()
    return pred_map, labels_sp


def evaluate_val_zeroshot(
    data_root,
    val_txt,
    model,
    processor,
    device,
    n_segments=64,
    compactness=10.0,
    min_size=100,
    context_margin=2,
):
    with open(val_txt, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    all_ious = []
    pixel_accs = []
    n_evaluated = 0
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 2:
            continue
        img_rel, mask_rel = parts[0], parts[1]
        img_path = os.path.join(data_root, img_rel)
        mask_path = os.path.join(data_root, mask_rel)
        if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
            print(f"  Skip (missing): {img_rel}")
            continue

        img = Image.open(img_path).convert("RGB")
        h, w = np.array(img).shape[:2]
        gt = load_gt_mask(mask_path, (h, w))

        pred_map, _ = run_medsiglip_superpixels_zeroshot(
            model,
            processor,
            img,
            device,
            n_segments=n_segments,
            compactness=compactness,
            min_size=min_size,
            context_margin=context_margin,
        )
        if pred_map is None:
            print(f"  Skip (no superpixels): {img_rel}")
            continue

        valid = gt != 255
        if valid.sum() == 0:
            continue

        ious = compute_iou_per_class(
            pred_map,
            gt,
            num_classes=NUM_CLASSES,
            ignore_index=255,
        )
        all_ious.append(ious)
        acc = (pred_map[valid] == gt[valid]).mean() * 100.0
        pixel_accs.append(acc)
        n_evaluated += 1
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(lines)}] {os.path.basename(img_rel)} pixel_acc={acc:.1f}%")

    if not all_ious:
        return None, None, None, 0
    all_ious = np.array(all_ious)
    mean_per_class = np.nanmean(all_ious, axis=0) * 100.0
    mean_miou = np.nanmean(mean_per_class)
    mean_pixel_acc = np.mean(pixel_accs)
    return mean_miou, mean_pixel_acc, mean_per_class, n_evaluated


def main():
    parser = argparse.ArgumentParser(
        description="MedSigLIP zero-shot superpixel baseline on LUTSeg val"
    )
    parser.add_argument("--data-root", default="data/LUTSeg", help="LUTSeg root")
    parser.add_argument("--val-txt", default="splits/lutseg/val.txt", help="Val split path")
    parser.add_argument("--n-segments", type=int, default=64)
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--min-size", type=int, default=100)
    parser.add_argument("--context-margin", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not HAS_SKIMAGE:
        print("Install scikit-image: pip install scikit-image")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = os.path.abspath(args.data_root)
    val_txt = os.path.abspath(args.val_txt)
    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        sys.exit(1)
    if not os.path.isfile(val_txt):
        print(f"Val split not found: {val_txt}")
        sys.exit(1)

    print("Loading MedSigLIP (google/medsiglip-448)...")
    model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    model.eval()

    print("\nEvaluating zero-shot on validation set...")
    mean_miou, mean_pixel_acc, mean_per_class, n_evaluated = evaluate_val_zeroshot(
        data_root,
        val_txt,
        model,
        processor,
        device,
        n_segments=args.n_segments,
        compactness=args.compactness,
        min_size=args.min_size,
        context_margin=args.context_margin,
    )
    if mean_miou is None:
        print("No valid images evaluated.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("MedSigLIP zero-shot on LUTSeg validation set")
    print("=" * 60)
    print(f"Images evaluated: {n_evaluated}")
    print(f"Pixel accuracy (mean): {mean_pixel_acc:.2f}%")
    print(f"mIoU ({NUM_CLASSES} classes): {mean_miou:.2f}%")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  IoU {name}: {mean_per_class[c]:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
