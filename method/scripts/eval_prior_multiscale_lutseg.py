#!/usr/bin/env python3
"""
Standalone check for multi-scale MedSigLIP classifier prior on LUTSeg.

Goal:
  Compare single-scale priors vs fused multi-scale prior before integrating
  into TiSage training.

It evaluates three prediction modes on val:
  1) coarse prior only
  2) fine prior only
  3) fused prior in logit space:
       z = beta * log(P_fine) + (1 - beta) * log(P_coarse)
       P_fused = softmax(z)

Usage:
  python eval_prior_multiscale_lutseg.py
  python scripts/eval_prior_multiscale_lutseg.py --classifier-path method/checkpoints/pretrained/medsiglip_head_lutseg.pt --beta 0.7
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
    extract_medsiglip_embeddings_batch,
    extract_superpixel_crops,
    load_classifier_checkpoint,
    load_gt_mask,
    segment_superpixels,
)


def compute_prior_single_scale(
    img_pil,
    model,
    processor,
    classifier,
    device,
    n_segments=64,
    compactness=10.0,
    min_size=100,
    context_margin=2,
    embed_batch_size=32,
    crop_mode="bbox",
    outside_fill="mean",
    small_region_ratio_thresh=0.0,
    small_region_zoom=1.0,
):
    """
    Return per-pixel prior map for one scale.
    Output: prior shape (C, H, W), probabilities sum to 1 over C.
    """
    arr = np.array(img_pil)
    H, W = arr.shape[:2]

    labels_sp = segment_superpixels(arr, n_segments=n_segments, compactness=compactness)
    crops, kept_ids = extract_superpixel_crops(
        arr,
        labels_sp,
        min_size=min_size,
        target_size=CROP_SIZE,
        preserve_aspect=True,
        context_margin=context_margin,
        crop_mode=crop_mode,
        outside_fill=outside_fill,
        small_region_ratio_thresh=small_region_ratio_thresh,
        small_region_zoom=small_region_zoom,
    )

    uniform = 1.0 / NUM_CLASSES
    if not crops:
        return np.full((NUM_CLASSES, H, W), uniform, dtype=np.float32)

    embs = extract_medsiglip_embeddings_batch(
        model,
        processor,
        crops,
        device,
        batch_size=embed_batch_size,
    )
    with torch.no_grad():
        x = torch.from_numpy(embs).float().to(device)
        logits = classifier(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()  # (N, C)

    id_to_idx = {lid: i for i, lid in enumerate(kept_ids)}
    prior_hwc = np.full((H, W, NUM_CLASSES), uniform, dtype=np.float32)
    for lid in np.unique(labels_sp):
        if lid in id_to_idx:
            prior_hwc[labels_sp == lid] = probs[id_to_idx[lid]]
    return prior_hwc.transpose(2, 0, 1)


def fuse_priors_logit_space(prior_coarse, prior_fine, beta=0.5, eps=1e-8):
    """Fuse priors in logit space and return probabilities (C, H, W)."""
    z_coarse = np.log(np.clip(prior_coarse, eps, 1.0))
    z_fine = np.log(np.clip(prior_fine, eps, 1.0))
    z = beta * z_fine + (1.0 - beta) * z_coarse
    z_t = torch.from_numpy(z).float()
    return F.softmax(z_t, dim=0).numpy().astype(np.float32)


def evaluate_modes(
    data_root,
    val_txt,
    model,
    processor,
    classifier,
    device,
    coarse_cfg,
    fine_cfg,
    beta=0.5,
):
    """
    Evaluate coarse / fine / fused on val set.
    Returns dict with metrics per mode.
    """
    with open(val_txt, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    modes = ("coarse", "fine", "fused")
    ious_by_mode = {m: [] for m in modes}
    accs_by_mode = {m: [] for m in modes}
    n_evaluated = 0

    classifier.eval()
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 2:
            continue
        img_rel, mask_rel = parts
        img_path = os.path.join(data_root, img_rel)
        mask_path = os.path.join(data_root, mask_rel)
        if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
            print(f"  Skip (missing): {img_rel}")
            continue

        img = Image.open(img_path).convert("RGB")
        H, W = np.array(img).shape[:2]
        gt = load_gt_mask(mask_path, (H, W))
        valid = gt != 255
        if valid.sum() == 0:
            continue

        prior_coarse = compute_prior_single_scale(
            img, model, processor, classifier, device, **coarse_cfg
        )
        prior_fine = compute_prior_single_scale(
            img, model, processor, classifier, device, **fine_cfg
        )
        prior_fused = fuse_priors_logit_space(prior_coarse, prior_fine, beta=beta)

        priors = {
            "coarse": prior_coarse,
            "fine": prior_fine,
            "fused": prior_fused,
        }
        for mode in modes:
            pred = priors[mode].argmax(axis=0).astype(np.int32)
            ious = compute_iou_per_class(pred, gt, num_classes=NUM_CLASSES, ignore_index=255)
            ious_by_mode[mode].append(ious)
            accs_by_mode[mode].append((pred[valid] == gt[valid]).mean() * 100.0)

        n_evaluated += 1
        if (idx + 1) % 5 == 0 or idx == 0:
            fused_acc = accs_by_mode["fused"][-1]
            print(f"  [{idx+1}/{len(lines)}] {os.path.basename(img_rel)} fused_pixel_acc={fused_acc:.1f}%")

    results = {}
    for mode in modes:
        if not ious_by_mode[mode]:
            results[mode] = None
            continue
        arr = np.array(ious_by_mode[mode])
        mean_per_class = np.nanmean(arr, axis=0) * 100.0
        mean_miou = np.nanmean(mean_per_class)
        mean_pixel_acc = float(np.mean(accs_by_mode[mode]))
        results[mode] = {
            "mIoU": float(mean_miou),
            "pixel_acc": mean_pixel_acc,
            "per_class_iou": mean_per_class,
            "n_images": n_evaluated,
        }
    return results


def print_report(mode_name, metrics):
    print("\n" + "=" * 68)
    print(f"{mode_name} prior on LUTSeg validation set")
    print("=" * 68)
    print(f"Images evaluated: {metrics['n_images']}")
    print(f"Pixel accuracy (mean): {metrics['pixel_acc']:.2f}%")
    print(f"mIoU ({NUM_CLASSES} classes): {metrics['mIoU']:.2f}%")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  IoU {name}: {metrics['per_class_iou'][c]:.2f}%")
    print("=" * 68)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate single vs multi-scale MedSigLIP classifier priors on LUTSeg val"
    )
    parser.add_argument("--data-root", default="data/LUTSeg")
    parser.add_argument("--val-txt", default="splits/lutseg/val.txt")
    parser.add_argument("--classifier-path", default="method/checkpoints/pretrained/medsiglip_head_lutseg.pt",
                        help="Path to saved LUTSeg classifier checkpoint (.pt)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--beta", type=float, default=0.5, help="Fusion weight for fine prior in logit space")
    parser.add_argument("--crop-mode", choices=["bbox", "masked_bbox", "masked_full"], default="bbox",
                        help="How each superpixel crop is formed before MedSigLIP")
    parser.add_argument("--outside-fill", choices=["mean", "gray", "black"], default="mean",
                        help="Fill value outside region for masked crop modes")
    parser.add_argument("--small-region-ratio-thresh", type=float, default=0.0,
                        help="For masked_full: apply zoom when region_area/image_area is below this threshold")
    parser.add_argument("--small-region-zoom", type=float, default=1.0,
                        help="For masked_full small regions: zoom-in factor (>1.0)")

    # Coarse superpixels (baseline-like)
    parser.add_argument("--coarse-n-segments", type=int, default=32)
    parser.add_argument("--coarse-compactness", type=float, default=10.0)
    parser.add_argument("--coarse-min-size", type=int, default=80)
    parser.add_argument("--coarse-context-margin", type=int, default=2)

    # Fine superpixels
    parser.add_argument("--fine-n-segments", type=int, default=128)
    parser.add_argument("--fine-compactness", type=float, default=10.0)
    parser.add_argument("--fine-min-size", type=int, default=40)
    parser.add_argument("--fine-context-margin", type=int, default=2)
    parser.add_argument("--results-tsv", default="", help="Append one row of metrics (and params) to this TSV for ablation scripts")
    parser.add_argument("--quiet", action="store_true", help="Less stdout when used in ablation loops")
    args = parser.parse_args()

    if not HAS_SKIMAGE:
        print("Install scikit-image: pip install scikit-image")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = os.path.abspath(args.data_root)
    val_txt = os.path.abspath(args.val_txt)
    classifier_path = os.path.abspath(args.classifier_path)

    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        sys.exit(1)
    if not os.path.isfile(val_txt):
        print(f"Val split not found: {val_txt}")
        sys.exit(1)
    if not os.path.isfile(classifier_path):
        print(f"Classifier checkpoint not found: {classifier_path}")
        sys.exit(1)
    if not (0.0 <= args.beta <= 1.0):
        print("--beta must be in [0, 1].")
        sys.exit(1)
    if args.small_region_zoom < 1.0:
        print("--small-region-zoom must be >= 1.0")
        sys.exit(1)
    if args.small_region_ratio_thresh < 0:
        print("--small-region-ratio-thresh must be >= 0")
        sys.exit(1)

    print("Loading MedSigLIP (google/medsiglip-448)...")
    model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    model.eval()

    print(f"Loading classifier: {classifier_path}")
    classifier, head_type, embed_dim = load_classifier_checkpoint(classifier_path)
    classifier = classifier.to(device)
    classifier.eval()
    print(f"  Loaded head_type={head_type}, embed_dim={embed_dim}")

    coarse_cfg = {
        "n_segments": args.coarse_n_segments,
        "compactness": args.coarse_compactness,
        "min_size": args.coarse_min_size,
        "context_margin": args.coarse_context_margin,
        "embed_batch_size": args.embed_batch_size,
        "crop_mode": args.crop_mode,
        "outside_fill": args.outside_fill,
        "small_region_ratio_thresh": args.small_region_ratio_thresh,
        "small_region_zoom": args.small_region_zoom,
    }
    fine_cfg = {
        "n_segments": args.fine_n_segments,
        "compactness": args.fine_compactness,
        "min_size": args.fine_min_size,
        "context_margin": args.fine_context_margin,
        "embed_batch_size": args.embed_batch_size,
        "crop_mode": args.crop_mode,
        "outside_fill": args.outside_fill,
        "small_region_ratio_thresh": args.small_region_ratio_thresh,
        "small_region_zoom": args.small_region_zoom,
    }

    if not args.quiet:
        print("\nEvaluating priors on validation set...")
        print(f"  Coarse cfg: {coarse_cfg}")
        print(f"  Fine cfg:   {fine_cfg}")
        print(f"  Fusion beta (fine weight): {args.beta}")

    results = evaluate_modes(
        data_root=data_root,
        val_txt=val_txt,
        model=model,
        processor=processor,
        classifier=classifier,
        device=device,
        coarse_cfg=coarse_cfg,
        fine_cfg=fine_cfg,
        beta=args.beta,
    )

    if not args.quiet:
        for mode in ("coarse", "fine", "fused"):
            if results[mode] is None:
                print(f"\nNo valid results for mode={mode}.")
                continue
            print_report(mode.capitalize(), results[mode])
        if results["coarse"] is not None and results["fused"] is not None:
            delta = results["fused"]["mIoU"] - results["coarse"]["mIoU"]
            print(f"\nDelta mIoU (fused - coarse): {delta:+.2f} points")

    if args.results_tsv:
        write_results_tsv(
            args.results_tsv,
            args.coarse_n_segments,
            args.coarse_min_size,
            args.fine_n_segments,
            args.fine_min_size,
            args.beta,
            args.crop_mode,
            args.outside_fill,
            args.small_region_ratio_thresh,
            args.small_region_zoom,
            results,
        )


def write_results_tsv(
    path,
    coarse_n,
    coarse_min,
    fine_n,
    fine_min,
    beta,
    crop_mode,
    outside_fill,
    small_region_ratio_thresh,
    small_region_zoom,
    results,
):
    """Append one row to path. Header written if file is new or empty."""
    row = {
        "coarse_n_segments": coarse_n,
        "coarse_min_size": coarse_min,
        "fine_n_segments": fine_n,
        "fine_min_size": fine_min,
        "beta": beta,
        "crop_mode": crop_mode,
        "outside_fill": outside_fill,
        "small_region_ratio_thresh": small_region_ratio_thresh,
        "small_region_zoom": small_region_zoom,
        "coarse_miou": results["coarse"]["mIoU"] if results.get("coarse") else "",
        "fine_miou": results["fine"]["mIoU"] if results.get("fine") else "",
        "fused_miou": results["fused"]["mIoU"] if results.get("fused") else "",
        "coarse_acc": results["coarse"]["pixel_acc"] if results.get("coarse") else "",
        "fine_acc": results["fine"]["pixel_acc"] if results.get("fine") else "",
        "fused_acc": results["fused"]["pixel_acc"] if results.get("fused") else "",
    }
    if results.get("fused") is not None:
        for c, name in enumerate(CLASS_NAMES):
            row[f"fused_iou_{name}"] = results["fused"]["per_class_iou"][c]

    cols = [
        "coarse_n_segments", "coarse_min_size", "fine_n_segments", "fine_min_size", "beta",
        "crop_mode", "outside_fill", "small_region_ratio_thresh", "small_region_zoom",
        "coarse_miou", "fine_miou", "fused_miou", "coarse_acc", "fine_acc", "fused_acc",
    ]
    cols += [f"fused_iou_{name}" for name in CLASS_NAMES]
    write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a") as f:
        if write_header:
            f.write("\t".join(cols) + "\n")
        vals = []
        for k in cols:
            v = row.get(k, "")
            vals.append(str(v) if v != "" else "")
        f.write("\t".join(vals) + "\n")


if __name__ == "__main__":
    main()
