#!/usr/bin/env python3
import argparse
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METHODS = ("deeplabv3plus", "dinov2_dpt", "fixmatch", "unimatch_v2", "tisage")
SPLITS = {
    "dfutissue": ("fixed", "1_4", "1_8", "1_16"),
    "lutseg": ("1_4", "1_8", "1_16", "full"),
}
SELECTED_SEEDS = {
    ("unimatch_v2", "dfutissue", "fixed"): 1,
    ("unimatch_v2", "dfutissue", "1_4"): 0,
    ("unimatch_v2", "dfutissue", "1_8"): 1,
    ("unimatch_v2", "dfutissue", "1_16"): 2,
    ("tisage", "dfutissue", "fixed"): 2,
    ("tisage", "dfutissue", "1_4"): 1,
    ("tisage", "dfutissue", "1_8"): 1,
    ("tisage", "dfutissue", "1_16"): 0,
}
TISAGE_PROFILES = {
    ("dfutissue", "fixed"): (0.2, 0.90, 0.50, 32, 128, 0.5, 5),
    ("dfutissue", "1_4"): (0.2, 0.90, 0.50, 32, 128, 0.5, 5),
    ("dfutissue", "1_8"): (0.2, 0.90, 0.50, 32, 128, 0.5, 5),
    ("dfutissue", "1_16"): (0.2, 0.90, 0.50, 32, 128, 0.5, 10),
    ("lutseg", "1_4"): (0.2, 0.85, 0.55, 64, 192, 0.7, 10),
    ("lutseg", "1_8"): (0.25, 0.85, 0.55, 64, 192, 0.7, 10),
    ("lutseg", "1_16"): (0.15, 0.90, 0.50, 64, 192, 0.7, 10),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one paper-reported Table 1 experiment")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=tuple(SPLITS), required=True)
    parser.add_argument("--split", choices=("fixed", "1_4", "1_8", "1_16", "full"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        help="worker count (paper default: 1 for DeepLabV3+, 2 otherwise)",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--epochs", type=int, help="override epochs for a bounded training check")
    parser.add_argument(
        "--max-steps-per-epoch",
        type=int,
        help="limit optimizer steps per epoch for a bounded training check",
    )
    parser.add_argument("--no-save", action="store_true", help="skip checkpoint files")
    parser.add_argument("--save-path", type=Path, help="override the experiment output directory")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    if args.split not in SPLITS[args.dataset]:
        raise ValueError(f"{args.dataset}/{args.split} is not reported in Table 1")
    if args.split == "full" and args.method not in {"deeplabv3plus", "dinov2_dpt"}:
        raise ValueError("LUTSeg full supervision is reported only for supervised baselines")

    split_key = "fixed" if args.split == "full" else args.split
    seed = args.seed if args.seed is not None else SELECTED_SEEDS.get(
        (args.method, args.dataset, args.split), 0
    )
    config_prefix = {
        "deeplabv3plus": "deeplabv3plus",
        "dinov2_dpt": "dinov2_dpt",
        "fixmatch": "tisage",
        "unimatch_v2": "tisage",
        "tisage": "tisage",
    }[args.method]
    config = ROOT / "method" / "configs" / f"{config_prefix}_{args.dataset}.yaml"
    script = {
        "deeplabv3plus": ROOT / "baselines" / "deeplabv3plus.py",
        "dinov2_dpt": ROOT / "supervised.py",
        "fixmatch": ROOT / "baselines" / "fixmatch.py",
        "unimatch_v2": ROOT / "baselines" / "unimatch_v2.py",
        "tisage": ROOT / "method" / "src" / "tisage" / "train.py",
    }[args.method]
    labeled = ROOT / "splits" / args.dataset / split_key / "labeled.txt"
    unlabeled = ROOT / "splits" / args.dataset / split_key / "unlabeled.txt"
    save_path = args.save_path or (
        ROOT / "exp" / "table1" / args.method / args.dataset / args.split / f"seed{seed}"
    )
    nproc_per_node = args.nproc_per_node
    if nproc_per_node is None:
        nproc_per_node = 1 if args.method == "deeplabv3plus" else 2
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        str(script),
        "--config",
        str(config),
        "--labeled-id-path",
        str(labeled),
        "--unlabeled-id-path",
        str(unlabeled),
        "--save-path",
        str(save_path),
        "--seed",
        str(seed),
    ]
    if args.deterministic:
        command.append("--deterministic")

    if args.method == "tisage":
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        if args.max_steps_per_epoch is not None:
            command.extend(["--max-steps-per-epoch", str(args.max_steps_per_epoch)])
        if args.no_save:
            command.append("--no-save")
        alpha, gate, soft_beta, coarse, fine, prior_beta, recompute = TISAGE_PROFILES[
            (args.dataset, args.split)
        ]
        classifier = ROOT / "method" / "checkpoints" / "pretrained" / f"medsiglip_head_{args.dataset}.pt"
        command.extend(
            [
                "--medsiglip",
                "--medsiglip-classifier-path",
                str(classifier),
                "--medsiglip-soft-label",
                "--medsiglip-use-multiscale",
                "--medsiglip-adaptive-alpha",
                "--medsiglip-conf-weighted-kl",
                "--medsiglip-alpha",
                str(alpha),
                "--medsiglip-gate-conf",
                str(gate),
                "--medsiglip-soft-beta",
                str(soft_beta),
                "--medsiglip-coarse-n-segments",
                str(coarse),
                "--medsiglip-coarse-min-size",
                "80",
                "--medsiglip-fine-n-segments",
                str(fine),
                "--medsiglip-fine-min-size",
                "40",
                "--medsiglip-prior-beta",
                str(prior_beta),
                "--medsiglip-recompute-every",
                str(recompute),
            ]
        )
    elif args.epochs is not None or args.max_steps_per_epoch is not None or args.no_save:
        raise ValueError("bounded training overrides are currently supported only for TiSage")
    return command


def main() -> None:
    args = parse_args()
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
