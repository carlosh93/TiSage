#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.semseg.dpt import DPT


MODEL_CONFIG = {
    "encoder_size": "base",
    "features": 128,
    "out_channels": [96, 192, 384, 768],
    "nclass": 6,
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
PALETTE = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (255, 255, 0),
    3: (255, 0, 255),
    4: (250, 144, 49),
    5: (0, 255, 255),
    255: (200, 200, 200),
}


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_ema" in checkpoint:
        state_dict = checkpoint["model_ema"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model = DPT(**MODEL_CONFIG)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@torch.inference_mode()
def predict(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = ((tensor - MEAN) / STD).to(device)
    height, width = image.shape[:2]
    resized = (int(height / 14 + 0.5) * 14, int(width / 14 + 0.5) * 14)
    if resized != (height, width):
        tensor = F.interpolate(tensor, size=resized, mode="bilinear", align_corners=True)
    logits = model(tensor)
    if logits.shape[-2:] != (height, width):
        logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)
    return logits.argmax(dim=1)[0].cpu().numpy()


def colorize(mask: np.ndarray) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        output[mask == class_id] = color
    return output


def load_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the paper's LUTSeg qualitative Figure 4")
    parser.add_argument("--data-root", type=Path, default=Path("data/LUTSeg"))
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--tisage-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "method/figures/figure4_cases.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "method/figures/figure4_lutseg_reproduced.png",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    baseline = load_model(args.baseline_checkpoint, device)
    tisage = load_model(args.tisage_checkpoint, device)
    cases = load_cases(args.cases)
    if len(cases) != 2:
        raise ValueError(f"Expected two Figure 4 cases, found {len(cases)}")

    figure, axes = plt.subplots(1, 8, figsize=(18, 4), dpi=220)
    for case_index, case in enumerate(cases):
        image = np.array(Image.open(args.data_root / case["image"]).convert("RGB"))
        mask = np.array(Image.open(args.data_root / case["mask"]))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape != image.shape[:2]:
            mask = np.array(
                Image.fromarray(mask.astype(np.uint8)).resize(
                    (image.shape[1], image.shape[0]), Image.Resampling.NEAREST
                )
            )
        panels = (
            ("Image", image),
            ("(a)", colorize(mask)),
            ("(b)", colorize(predict(baseline, image, device))),
            ("(c)", colorize(predict(tisage, image, device))),
        )
        for offset, (title, panel) in enumerate(panels):
            axis = axes[case_index * 4 + offset]
            axis.imshow(panel)
            axis.set_title(title)
            axis.axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, bbox_inches="tight")
    plt.close(figure)
    print(args.output)


if __name__ == "__main__":
    main()

