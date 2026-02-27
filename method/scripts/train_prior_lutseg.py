#!/usr/bin/env python3
"""
MedSigLIP embeddings + linear/MLP classifier for LUTSeg (6 classes) — standalone experiment.

Same flow as example_dermfound.py but using MedSigLIP (PyTorch only, no TensorFlow):
extract image embeddings from superpixel crops on labeled LUTSeg images,
train a linear or small MLP (embed_dim -> 6), then run the same pipeline on the val set
and report mIoU. Intended as a candidate prior for the TiSage pipeline.

Requires: torch, transformers, PIL, numpy, scikit-image (no TensorFlow).

Usage (from repo root):
  python train_prior_lutseg.py
  python train_prior_lutseg.py --epochs 30 --save-classifier method/checkpoints/pretrained/medsiglip_head_lutseg.pt
  python train_prior_lutseg.py --load-classifier method/checkpoints/pretrained/medsiglip_head_lutseg.pt --visualize
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

try:
    from skimage.segmentation import slic
    from skimage.measure import regionprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# LUTSeg: 6 classes, order = GT indices (same order as configs/lutseg.yaml prompts)
CLASS_NAMES = ["background", "epithelial", "slough", "granulation", "necrotic", "other"]
NUM_CLASSES = 6
CROP_SIZE = (448, 448)

# Canonical LUTSeg colors from data/LUTSeg/pipeline/common.py (BGR for OpenCV).
CLASS_COLORS_BGR = {
    0: (0, 0, 0),           # background black
    1: (0, 255, 0),         # epithelial green
    2: (0, 255, 255),       # slough yellow
    3: (255, 0, 255),       # granulation magenta
    4: (49, 144, 250),      # necrotic dark orange (#FA9031 in RGB)
    5: (255, 255, 0),       # other cyan
    255: (200, 200, 200),   # ignore light gray
}


def _bgr_to_rgb_hex(bgr):
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


CLASS_COLORS_HEX = [_bgr_to_rgb_hex(CLASS_COLORS_BGR[class_id]) for class_id in range(NUM_CLASSES)]
IGNORE_COLOR_HEX = _bgr_to_rgb_hex(CLASS_COLORS_BGR[255])


# ---------------------------------------------------------------------------
# Superpixel / crop / GT helpers (from example_medsiglip.py)
# ---------------------------------------------------------------------------

def load_gt_mask(path, target_size):
    """
    Load LUTSeg annotation. Returns (H, W) with values in {0..5} and 255 for ignore/invalid.
    target_size: (H, W) of the image to align with.
    """
    mask_img = Image.open(path)
    arr = np.array(mask_img)
    if arr.ndim == 3:
        arr = arr[:, :, 0] if arr.shape[2] >= 1 else arr.squeeze()
    arr = np.clip(arr.astype(np.int32), 0, 255)
    pil_mask = Image.fromarray(arr.astype(np.uint8))
    # PIL resize takes (width, height)
    pil_mask = pil_mask.resize((target_size[1], target_size[0]), Image.NEAREST)
    mask = np.array(pil_mask).astype(np.int32)
    invalid = (mask < 0) | ((mask >= NUM_CLASSES) & (mask != 255))
    mask[invalid] = 255
    return mask


def _crop_to_square_then_resize(crop_arr, target_size=(448, 448), pad_value=None):
    """Pad crop to square then resize. Same as example_medsiglip.py."""
    if crop_arr.dtype != np.uint8:
        crop_arr = np.clip(crop_arr, 0, 255).astype(np.uint8)
    h, w = crop_arr.shape[:2]
    side = max(h, w)
    if pad_value is None:
        pad_value = int(round(crop_arr.mean())) if crop_arr.ndim == 2 else tuple(int(round(x)) for x in crop_arr.mean(axis=(0, 1)))
    if crop_arr.ndim == 3:
        square = np.full((side, side, crop_arr.shape[2]), pad_value, dtype=crop_arr.dtype)
    else:
        square = np.full((side, side), pad_value, dtype=crop_arr.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    square[y0 : y0 + h, x0 : x0 + w] = crop_arr
    pil_square = Image.fromarray(square).resize(target_size, Image.BILINEAR)
    return pil_square


def _resolve_outside_fill(arr, outside_fill="mean"):
    """Return RGB fill tuple for masked areas."""
    mode = str(outside_fill).lower()
    if mode == "black":
        return (0, 0, 0)
    if mode == "gray":
        return (128, 128, 128)
    if mode == "mean":
        return tuple(int(round(x)) for x in arr.mean(axis=(0, 1)))
    raise ValueError(f"Unknown outside_fill: {outside_fill}")


def _zoom_crop_around_center(arr, center_row, center_col, zoom_factor):
    """Crop around (center_row, center_col) using zoom_factor (>1 means zoom-in)."""
    if zoom_factor is None or zoom_factor <= 1.0:
        return arr
    h, w = arr.shape[:2]
    crop_h = max(16, int(round(h / float(zoom_factor))))
    crop_w = max(16, int(round(w / float(zoom_factor))))
    cy = int(round(center_row))
    cx = int(round(center_col))
    top = max(0, cy - crop_h // 2)
    left = max(0, cx - crop_w // 2)
    bottom = min(h, top + crop_h)
    right = min(w, left + crop_w)
    top = max(0, bottom - crop_h)
    left = max(0, right - crop_w)
    return arr[top:bottom, left:right]


def segment_superpixels(arr, n_segments=64, compactness=10.0):
    """arr: (H, W, 3). Returns labels (H, W)."""
    if not HAS_SKIMAGE:
        raise ImportError("Install scikit-image: pip install scikit-image")
    try:
        labels = slic(arr, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)
    except TypeError:
        labels = slic(arr, n_segments=n_segments, compactness=compactness, start_label=0, multichannel=True)
    return labels


def extract_superpixel_crops(
    arr,
    labels,
    min_size=100,
    target_size=(448, 448),
    preserve_aspect=True,
    context_margin=2,
    crop_mode="bbox",
    outside_fill="mean",
    small_region_ratio_thresh=0.0,
    small_region_zoom=1.0,
):
    """
    Returns crops (list of PIL), kept_label_ids (list of int).
    crop_mode:
      - bbox: current behavior (bbox + context)
      - masked_bbox: bbox + context, but non-region pixels masked with outside_fill
      - masked_full: full image masked to region (optional zoom-in for small regions)
    """
    H, W = arr.shape[:2]
    img_area = float(max(1, H * W))
    regions = regionprops(labels)
    crops = []
    kept_label_ids = []
    crop_mode = str(crop_mode).lower()
    if crop_mode not in {"bbox", "masked_bbox", "masked_full"}:
        raise ValueError(f"Unknown crop_mode: {crop_mode}")
    global_fill = _resolve_outside_fill(arr, outside_fill=outside_fill)
    for r in regions:
        if r.area < min_size:
            continue
        kept_label_ids.append(r.label)
        region_mask_full = labels == r.label
        minr, minc, maxr, maxc = r.bbox
        if context_margin > 0:
            minr = max(0, minr - context_margin)
            minc = max(0, minc - context_margin)
            maxr = min(H, maxr + context_margin)
            maxc = min(W, maxc + context_margin)

        if crop_mode == "bbox":
            crop = arr[minr:maxr, minc:maxc]
        elif crop_mode == "masked_bbox":
            crop = arr[minr:maxr, minc:maxc].copy()
            region_mask_crop = region_mask_full[minr:maxr, minc:maxc]
            fill_val = _resolve_outside_fill(crop, outside_fill=outside_fill)
            crop[~region_mask_crop] = fill_val
        else:
            masked = np.full_like(arr, global_fill, dtype=np.uint8)
            masked[region_mask_full] = arr[region_mask_full]
            region_ratio = float(r.area) / img_area
            if (
                small_region_ratio_thresh is not None
                and small_region_ratio_thresh > 0
                and small_region_zoom is not None
                and small_region_zoom > 1.0
                and region_ratio < float(small_region_ratio_thresh)
            ):
                center_row, center_col = r.centroid
                masked = _zoom_crop_around_center(
                    masked, center_row=center_row, center_col=center_col, zoom_factor=small_region_zoom
                )
            crop = masked

        if preserve_aspect:
            pil_crop = _crop_to_square_then_resize(crop, target_size=target_size)
        else:
            pil_crop = Image.fromarray(crop).resize(target_size, Image.BILINEAR)
        crops.append(pil_crop)
    return crops, kept_label_ids


def compute_iou_per_class(pred, gt, num_classes=NUM_CLASSES, ignore_index=255):
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


# ---------------------------------------------------------------------------
# MedSigLIP embedding extraction (image-only, batched)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_medsiglip_embeddings_batch(model, processor, pil_crops, device, batch_size=32):
    """
    Run MedSigLIP image encoder on list of PIL crops (448x448). No text.
    Returns (N, D) numpy float32; D inferred from model output.
    """
    all_embs = []
    for i in range(0, len(pil_crops), batch_size):
        batch = pil_crops[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        outputs = model.get_image_features(**inputs)
        if hasattr(outputs, "pooler_output"):
            feats = outputs.pooler_output
        else:
            feats = outputs
        feats = F.normalize(feats, dim=-1)
        all_embs.append(feats.cpu().float().numpy())
    return np.concatenate(all_embs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Labeled-data loop: collect (embedding, label) from labeled split
# ---------------------------------------------------------------------------

def collect_labeled_embeddings(
    data_root,
    labeled_txt,
    model,
    processor,
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
    For each (image, mask) in labeled_txt: SLIC crops, majority GT per region, MedSigLIP embeddings.
    Returns (embeddings, labels) with shapes (N, D) and (N,) in {0..5}. D from model.
    Drops regions with no valid GT pixels.
    """
    if not HAS_SKIMAGE:
        raise RuntimeError("scikit-image required for superpixels")

    with open(labeled_txt, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    all_embs = []
    all_labels = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        img_rel, mask_rel = parts[0], parts[1]
        img_path = os.path.join(data_root, img_rel)
        mask_path = os.path.join(data_root, mask_rel)
        if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
            print(f"  Skip (missing file): {img_rel}")
            continue

        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        H, W = arr.shape[:2]
        gt = load_gt_mask(mask_path, (H, W))

        labels_sp = segment_superpixels(arr, n_segments=n_segments, compactness=compactness)
        crops, kept_ids = extract_superpixel_crops(
            arr, labels_sp, min_size=min_size, target_size=CROP_SIZE,
            preserve_aspect=True, context_margin=context_margin,
            crop_mode=crop_mode, outside_fill=outside_fill,
            small_region_ratio_thresh=small_region_ratio_thresh,
            small_region_zoom=small_region_zoom,
        )
        if not crops:
            continue

        # Majority GT per kept region; drop if no valid pixels
        region_labels = []
        kept_crop_indices = []
        for idx, lid in enumerate(kept_ids):
            mask = labels_sp == lid
            gt_region = gt[mask]
            gt_region = gt_region[gt_region != 255]
            if gt_region.size == 0:
                continue
            majority = int(np.bincount(gt_region.astype(int), minlength=NUM_CLASSES).argmax())
            region_labels.append(majority)
            kept_crop_indices.append(idx)

        if not region_labels:
            continue
        crops_kept = [crops[i] for i in kept_crop_indices]
        embs = extract_medsiglip_embeddings_batch(
            model, processor, crops_kept, device, batch_size=embed_batch_size
        )
        all_embs.append(embs)
        all_labels.extend(region_labels)

    if not all_embs:
        # No fixed embed dim for MedSigLIP; use placeholder, will exit before training
        return np.zeros((0, 1), dtype=np.float32), np.array([], dtype=np.int64)
    X = np.concatenate(all_embs, axis=0)
    y = np.array(all_labels, dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Classifier: linear or MLP embed_dim → 4
# ---------------------------------------------------------------------------

def build_classifier(head_type, embed_dim, num_classes=NUM_CLASSES, hidden=256):
    if head_type == "linear":
        return nn.Linear(embed_dim, num_classes)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )
    raise ValueError(f"Unknown head: {head_type}")


def compute_class_weights_balanced(y, num_classes=NUM_CLASSES):
    """Balanced class weights: n_samples / (n_classes * count_per_class). Rare classes get higher weight."""
    counts = np.bincount(y, minlength=num_classes)
    # avoid div by zero; unseen class gets weight 1.0
    weights = np.ones(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = len(y) / (num_classes * counts[c])
    return torch.from_numpy(weights).float()


def train_classifier(X, y, head_type="linear", epochs=30, lr=1e-3, batch_size=64, device=None, val_frac=0.1, class_weights=None):
    """Train classifier on (X, y). Returns trained module. class_weights: tensor of shape (num_classes,) or None."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(y)
    indices = np.random.permutation(n)
    if val_frac > 0 and n >= 20:
        nval = max(1, int(n * val_frac))
        val_idx = indices[:nval]
        train_idx = indices[nval:]
        X_val = torch.from_numpy(X[val_idx]).float().to(device)
        y_val = torch.from_numpy(y[val_idx]).long().to(device)
    else:
        train_idx = indices
        X_val = y_val = None

    X_train = torch.from_numpy(X[train_idx]).float()
    y_train = torch.from_numpy(y[train_idx]).long()
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    head = build_classifier(head_type, embed_dim=X.shape[1], num_classes=NUM_CLASSES)
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    head.train()
    for ep in range(epochs):
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = head(bx) if head_type == "linear" else head(bx)
            loss = criterion(logits, by)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep + 1) % 10 == 0 or epochs <= 15:
            print(f"  Epoch {ep+1}/{epochs} loss={total_loss/len(loader):.4f}")

    if X_val is not None:
        head.eval()
        with torch.no_grad():
            logits = head(X_val) if head_type == "linear" else head(X_val)
            acc = (logits.argmax(1) == y_val).float().mean().item() * 100.0
            print(f"  Val accuracy (held-out crops): {acc:.2f}%")
    return head


def load_classifier_checkpoint(path, fallback_head_type="linear"):
    """
    Load classifier from either:
      1) bundle dict: {'state_dict', 'embed_dim', 'head_type'}
      2) raw state_dict (backward compatible)
    Returns: (head_module, head_type, embed_dim)
    """
    payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        head_type = payload.get("head_type", fallback_head_type)
        embed_dim = payload.get("embed_dim", None)
    else:
        state_dict = payload
        head_type = fallback_head_type
        embed_dim = None

    # Backward-compatible inference for legacy checkpoints.
    if embed_dim is None:
        if "weight" in state_dict:
            head_type = "linear"
            embed_dim = int(state_dict["weight"].shape[1])
        elif "0.weight" in state_dict:
            head_type = "mlp"
            embed_dim = int(state_dict["0.weight"].shape[1])
        else:
            raise ValueError(f"Could not infer classifier architecture from checkpoint keys: {list(state_dict.keys())[:4]}")

    head = build_classifier(head_type, embed_dim=embed_dim, num_classes=NUM_CLASSES)
    head.load_state_dict(state_dict)
    return head, head_type, int(embed_dim)


# ---------------------------------------------------------------------------
# Val evaluation: SLIC + MedSigLIP + classifier → pred map → mIoU
# ---------------------------------------------------------------------------

def run_medsiglip_superpixels(
    model,
    processor,
    img_pil,
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
    """Return (embs (N,D), labels_sp (H,W), kept_ids) or (None, None, []) if no crops."""
    if not HAS_SKIMAGE:
        return None, None, []
    arr = np.array(img_pil)
    labels_sp = segment_superpixels(arr, n_segments=n_segments, compactness=compactness)
    crops, kept_ids = extract_superpixel_crops(
        arr, labels_sp, min_size=min_size, target_size=CROP_SIZE,
        preserve_aspect=True, context_margin=context_margin,
        crop_mode=crop_mode, outside_fill=outside_fill,
        small_region_ratio_thresh=small_region_ratio_thresh,
        small_region_zoom=small_region_zoom,
    )
    if not crops:
        return None, labels_sp, []
    embs = extract_medsiglip_embeddings_batch(model, processor, crops, device, batch_size=embed_batch_size)
    return embs, labels_sp, kept_ids


def evaluate_val(
    data_root,
    val_txt,
    model,
    processor,
    classifier,
    head_type,
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
    """Run MedSigLIP + classifier on val set; compute and return mIoU, pixel acc, per-class IoU."""
    with open(val_txt, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    all_ious = []
    pixel_accs = []
    n_evaluated = 0
    classifier.eval()
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
        H, W = np.array(img).shape[:2]
        gt = load_gt_mask(mask_path, (H, W))

        embs, labels_sp, kept_ids = run_medsiglip_superpixels(
            model, processor, img, device,
            n_segments=n_segments, compactness=compactness,
            min_size=min_size, context_margin=context_margin,
            embed_batch_size=embed_batch_size,
            crop_mode=crop_mode, outside_fill=outside_fill,
            small_region_ratio_thresh=small_region_ratio_thresh,
            small_region_zoom=small_region_zoom,
        )
        if embs is None or len(kept_ids) == 0:
            print(f"  Skip (no superpixels): {img_rel}")
            continue

        with torch.no_grad():
            x = torch.from_numpy(embs).float().to(device)
            logits = classifier(x) if head_type == "linear" else classifier(x)
            probs = F.softmax(logits, dim=1)
        probs_np = probs.cpu().numpy()
        pred_map = np.zeros_like(labels_sp, dtype=np.int32)
        id_to_idx = {lid: i for i, lid in enumerate(kept_ids)}
        for lid in np.unique(labels_sp):
            if lid in id_to_idx:
                pred_map[labels_sp == lid] = probs_np[id_to_idx[lid]].argmax()

        valid = gt != 255
        if valid.sum() == 0:
            continue
        ious = compute_iou_per_class(pred_map, gt, num_classes=NUM_CLASSES, ignore_index=255)
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


# ---------------------------------------------------------------------------
# Optional: visualization and class distribution
# ---------------------------------------------------------------------------

def print_class_distribution(y):
    """Print count and fraction per class."""
    print("\nLabeled crop class distribution:")
    for c in range(NUM_CLASSES):
        count = (y == c).sum()
        pct = count / len(y) * 100.0 if len(y) > 0 else 0.0
        print(f"  {CLASS_NAMES[c]:12s}: {count:5d} ({pct:.1f}%)")


def visualize_one_val_sample(
    data_root,
    val_txt,
    model,
    processor,
    classifier,
    head_type,
    device,
    save_path="medsiglip_classifier_val_viz_lutseg.png",
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
    """Run pipeline on first val image and save pred vs GT side-by-side."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("  matplotlib required for visualization; skipping.")
        return
    with open(val_txt, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return
    parts = lines[0].split()
    if len(parts) != 2:
        return
    img_path = os.path.join(data_root, parts[0])
    mask_path = os.path.join(data_root, parts[1])
    if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
        return
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]
    gt = load_gt_mask(mask_path, (H, W))
    embs, labels_sp, kept_ids = run_medsiglip_superpixels(
        model, processor, img, device,
        n_segments=n_segments, compactness=compactness,
        min_size=min_size, context_margin=context_margin,
        embed_batch_size=embed_batch_size,
        crop_mode=crop_mode, outside_fill=outside_fill,
        small_region_ratio_thresh=small_region_ratio_thresh,
        small_region_zoom=small_region_zoom,
    )
    if embs is None or len(kept_ids) == 0:
        return
    with torch.no_grad():
        x = torch.from_numpy(embs).float().to(device)
        logits = classifier(x) if head_type == "linear" else classifier(x)
        probs = F.softmax(logits, dim=1)
    probs_np = probs.cpu().numpy()
    pred_map = np.zeros_like(labels_sp, dtype=np.int32)
    id_to_idx = {lid: i for i, lid in enumerate(kept_ids)}
    for lid in np.unique(labels_sp):
        if lid in id_to_idx:
            pred_map[labels_sp == lid] = probs_np[id_to_idx[lid]].argmax()

    cmap = ListedColormap(CLASS_COLORS_HEX + [IGNORE_COLOR_HEX])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(arr)
    axes[0].set_title("Image")
    axes[0].axis("off")
    axes[1].imshow(pred_map, cmap=cmap, vmin=0, vmax=NUM_CLASSES)
    axes[1].set_title("MedSigLIP + classifier pred")
    axes[1].axis("off")
    mask_show = np.where(gt == 255, NUM_CLASSES, gt)
    axes[2].imshow(mask_show, cmap=cmap, vmin=0, vmax=NUM_CLASSES)
    axes[2].set_title("Ground truth")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved visualization to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MedSigLIP embeddings + classifier on LUTSeg (standalone experiment)"
    )
    parser.add_argument("--data-root", default="data/LUTSeg", help="LUTSeg root")
    parser.add_argument("--labeled-txt", default="splits/lutseg/fixed/labeled.txt",
                        help="Labeled image/mask pairs")
    parser.add_argument("--val-txt", default="splits/lutseg/val.txt",
                        help="Val image/mask pairs")
    parser.add_argument("--n-segments", type=int, default=None, help="SLIC segments (default from preset)")
    parser.add_argument("--compactness", type=float, default=10.0)
    parser.add_argument("--min-size", type=int, default=None, help="Min superpixel area (default from preset)")
    parser.add_argument("--context-margin", type=int, default=2)
    parser.add_argument("--crop-mode", choices=["bbox", "masked_bbox", "masked_full"], default="bbox",
                        help="How each superpixel crop is formed before MedSigLIP")
    parser.add_argument("--outside-fill", choices=["mean", "gray", "black"], default="mean",
                        help="Fill value outside region for masked crop modes")
    parser.add_argument("--small-region-ratio-thresh", type=float, default=0.0,
                        help="For masked_full: apply zoom when region_area/image_area is below this threshold")
    parser.add_argument("--small-region-zoom", type=float, default=1.0,
                        help="For masked_full small regions: zoom-in factor (>1.0)")
    parser.add_argument("--superpixel-preset", choices=["default", "fine", "finer"], default="default",
                        help="default: 64,100; fine: 128,50; finer: 200,40")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--embed-batch-size", type=int, default=32, help="MedSigLIP embedding batch size")
    parser.add_argument("--save-embeddings", default="", help="Save labeled embeddings to .npz")
    parser.add_argument("--load-embeddings", default="", help="Load embeddings from .npz (skip MedSigLIP)")
    parser.add_argument("--save-classifier", default="method/checkpoints/pretrained/medsiglip_head_lutseg.pt",
                        help="Save trained classifier state")
    parser.add_argument("--load-classifier", default="", help="Load classifier (skip training)")
    parser.add_argument("--visualize", action="store_true", help="Save one val image pred vs GT")
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="balanced",
                        help="balanced: upweight rare classes; none: uniform")
    parser.add_argument("--train-mixed", action="store_true",
                        help="Train on merged labeled crops from both coarse and fine SLIC (two scales)")
    parser.add_argument("--coarse-n-segments", type=int, default=64, help="Coarse scale n_segments (for --train-mixed)")
    parser.add_argument("--coarse-min-size", type=int, default=100, help="Coarse scale min_size (for --train-mixed)")
    parser.add_argument("--fine-n-segments", type=int, default=200, help="Fine scale n_segments (for --train-mixed)")
    parser.add_argument("--fine-min-size", type=int, default=40, help="Fine scale min_size (for --train-mixed)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Apply superpixel preset if not overridden
    if args.superpixel_preset == "fine":
        if args.n_segments is None:
            args.n_segments = 128
        if args.min_size is None:
            args.min_size = 50
        print("Using fine superpixel preset: n_segments=128, min_size=50")
    elif args.superpixel_preset == "finer":
        if args.n_segments is None:
            args.n_segments = 200
        if args.min_size is None:
            args.min_size = 40
        print("Using finer superpixel preset: n_segments=200, min_size=40")
    if args.n_segments is None:
        args.n_segments = 64
    if args.min_size is None:
        args.min_size = 100
    if args.small_region_zoom < 1.0:
        print("--small-region-zoom must be >= 1.0")
        sys.exit(1)
    if args.small_region_ratio_thresh < 0:
        print("--small-region-ratio-thresh must be >= 0")
        sys.exit(1)

    if not HAS_SKIMAGE:
        print("Install scikit-image: pip install scikit-image")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_root = os.path.abspath(args.data_root)
    labeled_txt = os.path.abspath(args.labeled_txt)
    val_txt = os.path.abspath(args.val_txt)
    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        sys.exit(1)
    if not os.path.isfile(labeled_txt) and not args.load_embeddings:
        print(f"Labeled split not found: {labeled_txt} (use --load-embeddings to skip)")
        sys.exit(1)
    if not os.path.isfile(val_txt):
        print(f"Val split not found: {val_txt}")
        sys.exit(1)
    print(
        "Crop extraction settings: "
        f"mode={args.crop_mode}, outside_fill={args.outside_fill}, "
        f"small_region_ratio_thresh={args.small_region_ratio_thresh}, "
        f"small_region_zoom={args.small_region_zoom}"
    )

    # 1) Get labeled embeddings (or load from cache)
    model = None
    processor = None
    if args.load_embeddings:
        print(f"Loading embeddings from {args.load_embeddings}...")
        data = np.load(args.load_embeddings, allow_pickle=True)
        X = data["embeddings"]
        y = data["labels"]
        print(f"  Loaded {len(y)} labeled crops.")
    else:
        print("Loading MedSigLIP (google/medsiglip-448)...")
        model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
        processor = AutoProcessor.from_pretrained("google/medsiglip-448")
        model.eval()
        if args.train_mixed:
            print("Collecting labeled embeddings (mixed: coarse + fine SLIC)...")
            X_c, y_c = collect_labeled_embeddings(
                data_root, labeled_txt, model, processor, device,
                n_segments=args.coarse_n_segments, compactness=args.compactness,
                min_size=args.coarse_min_size, context_margin=args.context_margin,
                embed_batch_size=args.embed_batch_size,
                crop_mode=args.crop_mode, outside_fill=args.outside_fill,
                small_region_ratio_thresh=args.small_region_ratio_thresh,
                small_region_zoom=args.small_region_zoom,
            )
            X_f, y_f = collect_labeled_embeddings(
                data_root, labeled_txt, model, processor, device,
                n_segments=args.fine_n_segments, compactness=args.compactness,
                min_size=args.fine_min_size, context_margin=args.context_margin,
                embed_batch_size=args.embed_batch_size,
                crop_mode=args.crop_mode, outside_fill=args.outside_fill,
                small_region_ratio_thresh=args.small_region_ratio_thresh,
                small_region_zoom=args.small_region_zoom,
            )
            if X_c.size == 0 and X_f.size == 0:
                X, y = np.zeros((0, 1), dtype=np.float32), np.array([], dtype=np.int64)
            elif X_c.size == 0:
                X, y = X_f, y_f
            elif X_f.size == 0:
                X, y = X_c, y_c
            else:
                X = np.vstack((X_c, X_f))
                y = np.concatenate((y_c, y_f))
            print(f"  Collected {len(y)} labeled crops (coarse: {len(y_c)}, fine: {len(y_f)}).")
        else:
            print("Collecting labeled embeddings (SLIC + MedSigLIP)...")
            X, y = collect_labeled_embeddings(
                data_root, labeled_txt, model, processor, device,
                n_segments=args.n_segments, compactness=args.compactness,
                min_size=args.min_size, context_margin=args.context_margin,
                embed_batch_size=args.embed_batch_size,
                crop_mode=args.crop_mode, outside_fill=args.outside_fill,
                small_region_ratio_thresh=args.small_region_ratio_thresh,
                small_region_zoom=args.small_region_zoom,
            )
            print(f"  Collected {len(y)} labeled crops.")
        if args.save_embeddings:
            save_dir = os.path.dirname(os.path.abspath(args.save_embeddings))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            np.savez(args.save_embeddings, embeddings=X, labels=y)
            print(f"  Saved to {args.save_embeddings}")
    print_class_distribution(y)

    if len(y) == 0:
        print("No labeled data. Exit.")
        sys.exit(1)

    # 2) Train or load classifier
    if args.load_classifier:
        print(f"Loading classifier from {args.load_classifier}...")
        head, loaded_head_type, loaded_embed_dim = load_classifier_checkpoint(
            args.load_classifier, fallback_head_type=args.head
        )
        if loaded_head_type != args.head:
            print(f"  Note: checkpoint head_type={loaded_head_type} (overrides --head={args.head}).")
            args.head = loaded_head_type
        if X.shape[1] > 1 and loaded_embed_dim != X.shape[1]:
            print(f"  Warning: checkpoint embed_dim={loaded_embed_dim} differs from loaded embeddings dim={X.shape[1]}.")
        head = head.to(device)
    else:
        class_weights = None
        if args.class_weights == "balanced":
            class_weights = compute_class_weights_balanced(y)
            print("Using balanced class weights for training.")
        print("Training classifier...")
        head = train_classifier(
            X, y, head_type=args.head, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=device, val_frac=0.1,
            class_weights=class_weights,
        )
        if args.save_classifier:
            save_dir = os.path.dirname(os.path.abspath(args.save_classifier))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    "state_dict": head.state_dict(),
                    "embed_dim": int(X.shape[1]),
                    "head_type": args.head,
                },
                args.save_classifier,
            )
            print(f"  Saved classifier to {args.save_classifier}")

    # MedSigLIP needed for val evaluation if we only loaded embeddings
    if model is None:
        print("Loading MedSigLIP for val evaluation...")
        model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
        processor = AutoProcessor.from_pretrained("google/medsiglip-448")
        model.eval()

    # 3) Val evaluation
    print("\nEvaluating on validation set...")
    mean_miou, mean_pixel_acc, mean_per_class, n_evaluated = evaluate_val(
        data_root, val_txt, model, processor, head, args.head, device,
        n_segments=args.n_segments, compactness=args.compactness,
        min_size=args.min_size, context_margin=args.context_margin,
        embed_batch_size=args.embed_batch_size,
        crop_mode=args.crop_mode, outside_fill=args.outside_fill,
        small_region_ratio_thresh=args.small_region_ratio_thresh,
        small_region_zoom=args.small_region_zoom,
    )
    if mean_miou is not None:
        print("\n" + "=" * 60)
        print("MedSigLIP + classifier on LUTSeg validation set")
        print("=" * 60)
        print(f"Images evaluated: {n_evaluated}")
        print(f"Pixel accuracy (mean): {mean_pixel_acc:.2f}%")
        print(f"mIoU ({NUM_CLASSES} classes): {mean_miou:.2f}%")
        for c, name in enumerate(CLASS_NAMES):
            print(f"  IoU {name}: {mean_per_class[c]:.2f}%")
        print("=" * 60)

    if args.visualize and mean_miou is not None:
        visualize_one_val_sample(
            data_root, val_txt, model, processor, head, args.head, device,
            save_path="medsiglip_classifier_val_viz_lutseg.png",
            n_segments=args.n_segments, compactness=args.compactness,
            min_size=args.min_size, context_margin=args.context_margin,
            embed_batch_size=args.embed_batch_size,
            crop_mode=args.crop_mode, outside_fill=args.outside_fill,
            small_region_ratio_thresh=args.small_region_ratio_thresh,
            small_region_zoom=args.small_region_zoom,
        )


if __name__ == "__main__":
    main()
