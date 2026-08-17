#!/usr/bin/env python3
import argparse
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("full", "no_multiscale", "no_adaptive_alpha", "no_confw_kl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reported DFUTissue 1/8 component ablation")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.nproc_per_node}",
        str(ROOT / "method/src/tisage/train.py"),
        "--config",
        str(ROOT / "method/configs/tisage_dfutissue_ablation.yaml"),
        "--labeled-id-path",
        str(ROOT / "splits/dfutissue/1_8/labeled.txt"),
        "--unlabeled-id-path",
        str(ROOT / "splits/dfutissue/1_8/unlabeled.txt"),
        "--save-path",
        str(ROOT / f"exp/component_ablation/dfutissue/1_8/{args.variant}/seed{args.seed}"),
        "--seed",
        str(args.seed),
        "--medsiglip",
        "--medsiglip-classifier-path",
        str(ROOT / "method/checkpoints/pretrained/medsiglip_head_dfutissue.pt"),
        "--medsiglip-soft-label",
        "--medsiglip-alpha",
        "0.2",
        "--medsiglip-gate-conf",
        "0.9",
        "--medsiglip-soft-beta",
        "0.5",
        "--medsiglip-coarse-n-segments",
        "32",
        "--medsiglip-coarse-min-size",
        "80",
        "--medsiglip-fine-n-segments",
        "128",
        "--medsiglip-fine-min-size",
        "40",
        "--medsiglip-prior-beta",
        "0.5",
        "--medsiglip-recompute-every",
        "5",
    ]
    if args.variant != "no_multiscale":
        command.append("--medsiglip-use-multiscale")
    if args.variant != "no_adaptive_alpha":
        command.append("--medsiglip-adaptive-alpha")
    if args.variant != "no_confw_kl":
        command.append("--medsiglip-conf-weighted-kl")

    print(shlex.join(command))
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
