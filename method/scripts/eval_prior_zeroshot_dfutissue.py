#!/usr/bin/env python3
"""
Run MedSigLIP zero-shot (superpixel) on DFUTissue validation set and report mIoU.
Use this to see how strong the prior is before relying on it in training.

Usage (from TiSage project root):
  python scripts/eval_medsiglip_on_val.py
  python scripts/eval_medsiglip_on_val.py --data-root data/DFUTissue --val-txt splits/dfutissue/val.txt
  python scripts/eval_medsiglip_on_val.py --n-segments 64 --context-margin 2 --min-size 100
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Add project root for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from skimage.segmentation import slic
    from skimage.measure import regionprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Same 4-class prompts as calibrator (order = GT indices)
TISSUE_TEXTS = [
    "a photo of healthy skin with no wound",
    "a photo of fibrin tissue in a wound",
    "a photo of red granulation tissue in a wound",
    "a photo of callus tissue around an ulcer",
]
CLASS_NAMES = ["background", "fibrin", "granulation", "callus"]
NUM_CLASSES = 4


def load_gt_mask(path, target_size):
    """Load annotation; resize to target_size (H, W) with NEAREST. Returns (H, W) int32, 255 = ignore."""
    mask_img = Image.open(path)
    arr = np.array(mask_img)
    if arr.ndim == 3:
        arr = arr[:, :, 0] if arr.shape[2] >= 1 else arr.squeeze()
    arr = np.clip(arr.astype(np.int32), 0, 255)
    pil_mask = Image.fromarray(arr.astype(np.uint8))
    pil_mask = pil_mask.resize((target_size[1], target_size[0]), Image.NEAREST)
    return np.array(pil_mask).astype(np.int32)


def _crop_to_square_then_resize(crop_arr, target_size=(448, 448)):
    h, w = crop_arr.shape[:2]
    side = max(h, w)
    pad_val = tuple(int(round(x)) for x in crop_arr.mean(axis=(0, 1)))
    square = np.full((side, side, 3), pad_val, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    square[y0 : y0 + h, x0 : x0 + w] = crop_arr
    return Image.fromarray(square).resize(target_size, Image.BILINEAR)


def segment_superpixels(arr, n_segments=64, compactness=10.0):
    try:
        labels = slic(arr, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)
    except TypeError:
        labels = slic(arr, n_segments=n_segments, compactness=compactness, start_label=0, multichannel=True)
    return labels


def extract_superpixel_crops(arr, labels, min_size=100, context_margin=2):
    H, W = arr.shape[:2]
    regions = regionprops(labels)
    crops, kept_ids = [], []
    for r in regions:
        if r.area < min_size:
            continue
        kept_ids.append(r.label)
        minr, minc, maxr, maxc = r.bbox
        if context_margin > 0:
            minr = max(0, minr - context_margin)
            minc = max(0, minc - context_margin)
            maxr = min(H, maxr + context_margin)
            maxc = min(W, maxc + context_margin)
        crop = arr[minr:maxr, minc:maxc]
        crops.append(_crop_to_square_then_resize(crop))
    return crops, kept_ids


def run_medsiglip_superpixels(model, processor, img_pil, n_segments=64, compactness=10.0, min_size=100, context_margin=2):
    """Return (pred_map (H,W), labels (H,W), probs (N,4), kept_ids) or (None,...) if no crops."""
    if not HAS_SKIMAGE:
        return None, None, None, None
    arr = np.array(img_pil)
    labels = segment_superpixels(arr, n_segments=n_segments, compactness=compactness)
    crops, kept_ids = extract_superpixel_crops(arr, labels, min_size=min_size, context_margin=context_margin)
    if not crops:
        return None, labels, None, []
    inputs = processor(text=TISSUE_TEXTS, images=crops, padding="max_length", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs)
        probs = F.softmax(out.logits_per_image, dim=1)
    probs_np = probs.cpu().numpy()
    pred_map = np.zeros_like(labels, dtype=np.int32)
    id_to_idx = {lid: i for i, lid in enumerate(kept_ids)}
    for lid in np.unique(labels):
        if lid in id_to_idx:
            pred_map[labels == lid] = probs_np[id_to_idx[lid]].argmax()
    return pred_map, labels, probs_np, kept_ids


def compute_iou_per_class(pred, gt, num_classes=4, ignore_index=255):
    valid = gt != ignore_index
    pred_flat = pred[valid]
    gt_flat = gt[valid]
    if gt_flat.size == 0:
        return [float("nan")] * num_classes
    ious = []
    for c in range(num_classes):
        inter = ((pred_flat == c) & (gt_flat == c)).sum()
        union = ((pred_flat == c) | (gt_flat == c)).sum()
        ious.append(float(inter) / union if union > 0 else float("nan"))
    return ious


def main():
    parser = argparse.ArgumentParser(description="MedSigLIP zero-shot mIoU on DFUTissue val set")
    parser.add_argument("--data-root", default="data/DFUTissue", help="DFUTissue root")
    parser.add_argument("--val-txt", default="splits/dfutissue/val.txt", help="Path to val.txt (image mask pairs)")
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
    print("Loading MedSigLIP (google/medsiglip-448)...")
    model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
    processor = AutoProcessor.from_pretrained("google/medsiglip-448")
    model.eval()

    val_path = args.val_txt
    if not os.path.isabs(val_path):
        val_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), val_path)
    data_root = args.data_root
    if not os.path.isabs(data_root):
        data_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), data_root)

    with open(val_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    all_ious = []
    pixel_accs = []
    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 2:
            continue
        img_rel, mask_rel = parts[0], parts[1]
        img_path = os.path.join(data_root, img_rel)
        mask_path = os.path.join(data_root, mask_rel)
        if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
            print(f"Skip (missing file): {img_rel}")
            continue
        img = Image.open(img_path).convert("RGB")
        H, W = np.array(img).shape[:2]
        gt = load_gt_mask(mask_path, (H, W))

        pred_map, _, _, _ = run_medsiglip_superpixels(
            model, processor, img,
            n_segments=args.n_segments,
            compactness=args.compactness,
            min_size=args.min_size,
            context_margin=args.context_margin,
        )
        if pred_map is None:
            print(f"Skip (no superpixels): {img_rel}")
            continue

        valid = gt != 255
        if valid.sum() == 0:
            continue
        ious = compute_iou_per_class(pred_map, gt, num_classes=NUM_CLASSES, ignore_index=255)
        all_ious.append(ious)
        acc = (pred_map[valid] == gt[valid]).mean() * 100.0
        pixel_accs.append(acc)
        if (idx + 1) % 5 == 0 or idx == 0:
            print(f"  [{idx+1}/{len(lines)}] {os.path.basename(img_rel)} pixel_acc={acc:.1f}%")

    if not all_ious:
        print("No valid images evaluated.")
        sys.exit(1)

    # Aggregate
    all_ious = np.array(all_ious)
    mean_per_class = np.nanmean(all_ious, axis=0) * 100.0
    mean_miou = np.nanmean(mean_per_class)
    mean_pixel_acc = np.mean(pixel_accs)

    print("\n" + "=" * 60)
    print("MedSigLIP zero-shot on DFUTissue validation set")
    print("=" * 60)
    print(f"Images evaluated: {len(all_ious)}")
    print(f"Pixel accuracy (mean): {mean_pixel_acc:.2f}%")
    print(f"mIoU (4 classes): {mean_miou:.2f}%")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  IoU {name}: {mean_per_class[c]:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
