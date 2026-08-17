#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from util.classes import CLASSES
from util.utils import intersectionAndUnion


MODEL_CONFIGS = {
    "small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
    "base": {"encoder_size": "base", "features": 128, "out_channels": [96, 192, 384, 768]},
    "large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "giant": {"encoder_size": "giant", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TiSage or UniMatch-V2 DPT checkpoint")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", choices=("model", "model_ema"), default="model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate checkpoint loading without running inference",
    )
    return parser.parse_args()


def load_state(checkpoint_path: Path, requested_state: str) -> tuple[dict[str, torch.Tensor], str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_name = requested_state
    if state_name not in checkpoint:
        raise KeyError(f"Checkpoint has no {state_name!r} state")
    state = {
        key.removeprefix("module."): value
        for key, value in checkpoint[state_name].items()
    }
    return state, state_name


def build_model(config: dict[str, object], checkpoint: Path, state: str) -> tuple[torch.nn.Module, str]:
    backbone_size = str(config["backbone"]).split("_")[-1]
    model = DPT(**{**MODEL_CONFIGS[backbone_size], "nclass": int(config["nclass"])})
    state_dict, state_name = load_state(checkpoint, state)
    model.load_state_dict(state_dict)
    return model, state_name


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    intersection_sum = np.zeros(num_classes, dtype=np.float64)
    union_sum = np.zeros(num_classes, dtype=np.float64)
    model.eval()
    for image, mask, _ in loader:
        original_size = image.shape[-2:]
        resized = tuple(int(size / 14 + 0.5) * 14 for size in original_size)
        image = image.to(device)
        if resized != original_size:
            image = F.interpolate(image, resized, mode="bilinear", align_corners=True)
        logits = model(image)
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, original_size, mode="bilinear", align_corners=True)
        prediction = logits.argmax(dim=1).cpu().numpy()
        intersection, union, _ = intersectionAndUnion(
            prediction, mask.numpy(), num_classes, 255
        )
        intersection_sum += intersection
        union_sum += union
    iou = intersection_sum / np.maximum(union_sum, 1e-10) * 100.0
    dice = 2.0 * intersection_sum / np.maximum(union_sum + intersection_sum, 1e-10) * 100.0
    return iou, dice


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    dataset = str(config["dataset"])
    model, state_name = build_model(config, args.checkpoint, args.state)
    if args.check_only:
        print(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "state": state_name,
                    "dataset": dataset,
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "status": "ok",
                },
                indent=2,
            )
        )
        return
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit(
            "Quantitative DINOv2 evaluation requires a CUDA GPU. "
            "Use --check-only to validate a checkpoint without inference."
        )
    model = model.to(device)
    validation = SemiDataset(dataset, str(config["data_root"]), "val")
    loader = DataLoader(
        validation,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    iou, dice = evaluate(model, loader, device, int(config["nclass"]))
    result = {
        "checkpoint": str(args.checkpoint),
        "state": state_name,
        "dataset": dataset,
        "samples": len(validation),
        "mIoU": round(float(iou.mean()), 4),
        "mean_dice": round(float(dice.mean()), 4),
        "classes": {
            name: {"iou": round(float(class_iou), 4), "dice": round(float(class_dice), 4)}
            for name, class_iou, class_dice in zip(CLASSES[dataset], iou, dice)
        },
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
