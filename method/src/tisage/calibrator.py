"""
MedSigLIP semantic calibrator for TiSage.

Provides per-pixel class priors from MedSigLIP (frozen, zero-shot or
embedding-classifier) to reweight the teacher's pseudo-labels during
semi-supervised training.

Usage (inside TiSage training loop):
    calibrator = MedSigLIPCalibrator(device, ...)
    medsiglip_prior = calibrator.compute_pixel_prior(img_u_w)  # (B, C, H, W)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.segmentation import slic
from skimage.measure import regionprops
from transformers import AutoModel, AutoProcessor

# ImageNet normalization used by the TiSage training pipeline (dataset/transform.py)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default prompts by dataset (order must match mask class indices).
DEFAULT_TEXT_PROMPTS = {
    "dfutissue": [
        "a photo of healthy skin with no wound",
        "a photo of fibrin tissue in a wound",
        "a photo of red granulation tissue in a wound",
        "a photo of callus tissue around an ulcer",
    ],
    "lutseg": [
        "a photo of healthy skin with no wound",
        "a photo of epithelial tissue in a wound",
        "a photo of slough tissue in a wound",
        "a photo of granulation tissue in a wound",
        "a photo of necrotic tissue in a wound",
        "a photo of other tissue in a wound",
    ],
}


def _crop_to_square_then_resize(crop_arr, target_size=(448, 448)):
    """Pad crop to square (mean-fill), then resize to target_size."""
    if crop_arr.dtype != np.uint8:
        crop_arr = np.clip(crop_arr, 0, 255).astype(np.uint8)
    h, w = crop_arr.shape[:2]
    side = max(h, w)
    pad_val = tuple(int(round(x)) for x in crop_arr.mean(axis=(0, 1)))
    square = np.full((side, side, 3), pad_val, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    square[y0 : y0 + h, x0 : x0 + w] = crop_arr
    return Image.fromarray(square).resize(target_size, Image.BILINEAR)


def _build_classifier_head(head_type, embed_dim, num_classes, hidden=256):
    if head_type == "linear":
        return nn.Linear(embed_dim, num_classes)
    if head_type == "mlp":
        return nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
        )
    raise ValueError(f"Unknown MedSigLIP classifier head_type: {head_type}")


class MedSigLIPCalibrator:
    """
    Frozen MedSigLIP model that produces per-pixel semantic priors for
    wound tissue classes via superpixel-level region classification.
    """

    def __init__(
        self,
        device,
        n_segments=64,
        compactness=10.0,
        min_size=100,
        context_margin=2,
        slic_seed=42,
        classifier_path=None,
        embed_batch_size=32,
        use_multiscale=False,
        coarse_n_segments=None,
        coarse_min_size=None,
        fine_n_segments=None,
        fine_min_size=None,
        prior_beta=0.5,
        dataset_name="dfutissue",
        num_classes=None,
        class_prompts=None,
    ):
        self.device = device
        self.dataset_name = str(dataset_name).lower()
        self.n_segments = n_segments
        self.compactness = compactness
        self.min_size = min_size
        self.context_margin = context_margin
        self.slic_seed = slic_seed
        self.embed_batch_size = embed_batch_size
        self.classifier = None
        self.use_multiscale = use_multiscale
        self.coarse_n_segments = coarse_n_segments if coarse_n_segments is not None else n_segments
        self.coarse_min_size = coarse_min_size if coarse_min_size is not None else min_size
        self.fine_n_segments = fine_n_segments if fine_n_segments is not None else n_segments
        self.fine_min_size = fine_min_size if fine_min_size is not None else min_size
        self.prior_beta = float(np.clip(prior_beta, 0.0, 1.0))
        self.num_classes = int(num_classes) if num_classes is not None else None
        if self.num_classes is not None and self.num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {self.num_classes}")

        if class_prompts is not None:
            self.texts = [str(p) for p in class_prompts]
        else:
            self.texts = list(DEFAULT_TEXT_PROMPTS.get(self.dataset_name, []))

        # Load frozen MedSigLIP
        self.model = AutoModel.from_pretrained("google/medsiglip-448").to(device)
        self.processor = AutoProcessor.from_pretrained("google/medsiglip-448")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if classifier_path:
            ckpt = torch.load(classifier_path, map_location="cpu")
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
                head_type = ckpt.get("head_type", "linear")
                embed_dim = ckpt.get("embed_dim", None)
            else:
                state_dict = ckpt
                head_type = "linear"
                embed_dim = None

            # Backward-compatible inference for raw state_dict checkpoints.
            inferred_num_classes = None
            if embed_dim is None:
                if "weight" in state_dict:
                    head_type = "linear"
                    embed_dim = int(state_dict["weight"].shape[1])
                    inferred_num_classes = int(state_dict["weight"].shape[0])
                elif "0.weight" in state_dict:
                    head_type = "mlp"
                    embed_dim = int(state_dict["0.weight"].shape[1])
                    if "2.weight" in state_dict:
                        inferred_num_classes = int(state_dict["2.weight"].shape[0])
                else:
                    raise ValueError(
                        f"Could not infer classifier architecture from checkpoint keys: {list(state_dict.keys())[:4]}"
                    )
            elif head_type == "linear" and "weight" in state_dict:
                inferred_num_classes = int(state_dict["weight"].shape[0])
            elif head_type == "mlp" and "2.weight" in state_dict:
                inferred_num_classes = int(state_dict["2.weight"].shape[0])

            if self.num_classes is None:
                if inferred_num_classes is None:
                    raise ValueError(
                        "num_classes was not provided and could not be inferred from classifier checkpoint."
                    )
                self.num_classes = inferred_num_classes
            elif inferred_num_classes is not None and self.num_classes != inferred_num_classes:
                raise ValueError(
                    f"num_classes ({self.num_classes}) does not match classifier output ({inferred_num_classes})."
                )

            self.classifier = _build_classifier_head(
                head_type, int(embed_dim), self.num_classes
            ).to(device)
            self.classifier.load_state_dict(state_dict)
            self.classifier.eval()
            for p in self.classifier.parameters():
                p.requires_grad = False
        else:
            if not self.texts:
                raise ValueError(
                    f"No prompts found for dataset '{self.dataset_name}'. "
                    "Provide class_prompts in config or use a classifier checkpoint."
                )
            if self.num_classes is None:
                self.num_classes = len(self.texts)
            elif self.num_classes != len(self.texts):
                raise ValueError(
                    f"num_classes ({self.num_classes}) does not match number of prompts ({len(self.texts)})."
                )

        if self.texts and len(self.texts) != self.num_classes:
            raise ValueError(
                f"Prompt count ({len(self.texts)}) must equal num_classes ({self.num_classes})."
            )

    # ------------------------------------------------------------------
    # Denormalize ImageNet-normalized tensor -> uint8 PIL
    # ------------------------------------------------------------------
    @staticmethod
    def _denormalize(img_tensor):
        """
        img_tensor: (3, H, W) float tensor, ImageNet-normalized.
        Returns: PIL Image (RGB, uint8).
        """
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
        img = img * IMAGENET_STD + IMAGENET_MEAN
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    # ------------------------------------------------------------------
    # SLIC superpixels (deterministic via fixed seed)
    # ------------------------------------------------------------------
    def _segment(self, arr, n_segments=None):
        """Run SLIC on (H, W, 3) uint8 array. Returns label map (H, W)."""
        if n_segments is None:
            n_segments = self.n_segments
        rng_state = np.random.get_state()
        np.random.seed(self.slic_seed)
        try:
            labels = slic(
                arr,
                n_segments=n_segments,
                compactness=self.compactness,
                start_label=0,
                channel_axis=-1,
            )
        except TypeError:
            labels = slic(
                arr,
                n_segments=n_segments,
                compactness=self.compactness,
                start_label=0,
                multichannel=True,
            )
        np.random.set_state(rng_state)
        return labels

    # ------------------------------------------------------------------
    # Extract crops per region
    # ------------------------------------------------------------------
    def _extract_crops(self, arr, labels, min_size=None):
        """
        arr: (H, W, 3) uint8.
        labels: (H, W) int from SLIC.
        Returns: crops (list[PIL]), kept_ids (list[int]).
        """
        if min_size is None:
            min_size = self.min_size
        H, W = arr.shape[:2]
        regions = regionprops(labels)
        crops, kept_ids = [], []
        for r in regions:
            if r.area < min_size:
                continue
            kept_ids.append(r.label)
            minr, minc, maxr, maxc = r.bbox
            if self.context_margin > 0:
                minr = max(0, minr - self.context_margin)
                minc = max(0, minc - self.context_margin)
                maxr = min(H, maxr + self.context_margin)
                maxc = min(W, maxc + self.context_margin)
            crop = arr[minr:maxr, minc:maxc]
            crops.append(_crop_to_square_then_resize(crop))
        return crops, kept_ids

    # ------------------------------------------------------------------
    # MedSigLIP image embeddings (image-only path for trained classifier)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _get_image_embeddings(self, crops):
        inputs = self.processor(images=crops, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        return F.normalize(feats, dim=-1)

    # ------------------------------------------------------------------
    # Region classification (classifier path or zero-shot path)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _classify_crops(self, crops):
        """
        crops: list of PIL Images.
        Returns: probs (N, C) tensor on self.device.
        """
        if self.classifier is not None:
            all_probs = []
            for i in range(0, len(crops), self.embed_batch_size):
                batch = crops[i : i + self.embed_batch_size]
                feats = self._get_image_embeddings(batch)
                logits = self.classifier(feats)
                all_probs.append(F.softmax(logits, dim=1))
            return torch.cat(all_probs, dim=0)

        if not self.texts:
            raise RuntimeError("Zero-shot classification requires class prompts.")

        inputs = self.processor(
            text=self.texts,
            images=crops,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image  # (N, C)
        return F.softmax(logits, dim=1)

    # ------------------------------------------------------------------
    # Build per-pixel prior for one image (single scale)
    # ------------------------------------------------------------------
    def _build_pixel_prior_single_scale(self, arr, n_segments, min_size):
        """
        arr: (H, W, 3) uint8 numpy array.
        Returns: prior (C, H, W) float32 numpy array.
        """
        H, W = arr.shape[:2]
        labels = self._segment(arr, n_segments=n_segments)
        crops, kept_ids = self._extract_crops(arr, labels, min_size=min_size)

        # Uniform prior for images with no valid superpixels
        uniform = 1.0 / self.num_classes
        if not crops:
            return np.full((self.num_classes, H, W), uniform, dtype=np.float32)

        probs = self._classify_crops(crops).cpu().numpy()  # (N, C)

        # Map region_id -> index in probs
        id_to_idx = {lid: i for i, lid in enumerate(kept_ids)}

        # Broadcast region probs to pixels
        prior = np.full((H, W, self.num_classes), uniform, dtype=np.float32)
        for lid in np.unique(labels):
            if lid in id_to_idx:
                prior[labels == lid] = probs[id_to_idx[lid]]

        return prior.transpose(2, 0, 1)  # (C, H, W)

    # ------------------------------------------------------------------
    # Build per-pixel prior for one image (single or multiscale)
    # ------------------------------------------------------------------
    def _build_pixel_prior(self, arr):
        if not self.use_multiscale:
            return self._build_pixel_prior_single_scale(
                arr,
                n_segments=self.n_segments,
                min_size=self.min_size,
            )

        prior_coarse = self._build_pixel_prior_single_scale(
            arr,
            n_segments=self.coarse_n_segments,
            min_size=self.coarse_min_size,
        )
        prior_fine = self._build_pixel_prior_single_scale(
            arr,
            n_segments=self.fine_n_segments,
            min_size=self.fine_min_size,
        )
        eps = 1e-8
        z_coarse = np.log(np.clip(prior_coarse, eps, 1.0))
        z_fine = np.log(np.clip(prior_fine, eps, 1.0))
        z_prior = self.prior_beta * z_fine + (1.0 - self.prior_beta) * z_coarse
        return F.softmax(torch.from_numpy(z_prior).float(), dim=0).numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Main entry point: batch of images -> (B, C, H, W) prior tensor
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_pixel_prior(self, img_u_w_batch):
        """
        img_u_w_batch: (B, 3, H, W) ImageNet-normalized tensor (weakly augmented).
        Returns: (B, C, H, W) float tensor on same device, softmax-like priors per pixel.
        """
        B = img_u_w_batch.shape[0]
        priors = []
        for b in range(B):
            pil_img = self._denormalize(img_u_w_batch[b])
            arr = np.array(pil_img)
            prior = self._build_pixel_prior(arr)  # (C, H, W)
            priors.append(prior)
        prior_np = np.stack(priors, axis=0)  # (B, C, H, W)
        return torch.from_numpy(prior_np).to(img_u_w_batch.device)
