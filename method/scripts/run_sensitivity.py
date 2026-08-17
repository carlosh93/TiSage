#!/usr/bin/env python3
import argparse
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VALUES = {
    "gate": (0.80, 0.85, 0.90, 0.95),
    "alpha": (0.10, 0.15, 0.20, 0.25, 0.30),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reported LUTSeg 1/8 sensitivity study")
    parser.add_argument("--parameter", choices=tuple(VALUES), required=True)
    parser.add_argument("--value", type=float, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not any(abs(args.value - value) < 1e-8 for value in VALUES[args.parameter]):
        raise ValueError(f"Unreported {args.parameter} value: {args.value}")

    gate = args.value if args.parameter == "gate" else 0.90
    alpha = args.value if args.parameter == "alpha" else 0.20
    run_name = f"{args.parameter}_{args.value:.2f}"
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(ROOT / "method/src/tisage/train.py"),
        "--config",
        str(ROOT / "method/configs/tisage_lutseg.yaml"),
        "--labeled-id-path",
        str(ROOT / "splits/lutseg/1_8/labeled.txt"),
        "--unlabeled-id-path",
        str(ROOT / "splits/lutseg/1_8/unlabeled.txt"),
        "--save-path",
        str(ROOT / f"exp/sensitivity/lutseg/1_8/{run_name}/seed0"),
        "--seed",
        "0",
        "--medsiglip",
        "--medsiglip-classifier-path",
        str(ROOT / "method/checkpoints/pretrained/medsiglip_head_lutseg.pt"),
        "--medsiglip-soft-label",
        "--medsiglip-use-multiscale",
        "--medsiglip-adaptive-alpha",
        "--medsiglip-conf-weighted-kl",
        "--medsiglip-alpha",
        str(alpha),
        "--medsiglip-gate-conf",
        str(gate),
        "--medsiglip-soft-beta",
        "0.5",
        "--medsiglip-coarse-n-segments",
        "64",
        "--medsiglip-coarse-min-size",
        "80",
        "--medsiglip-fine-n-segments",
        "192",
        "--medsiglip-fine-min-size",
        "40",
        "--medsiglip-prior-beta",
        "0.7",
        "--medsiglip-recompute-every",
        "5",
    ]
    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
