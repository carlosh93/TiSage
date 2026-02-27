#!/usr/bin/env python3
import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CLASS_RE = re.compile(
    r"Class \[(\d+) ([^\]]+)\] IoU:\s*([0-9.]+)(?:,\s*EMA:\s*([0-9.]+))?"
)
MEAN_RE = re.compile(r"MeanIoU:\s*([0-9.]+)(?:,\s*EMA:\s*([0-9.]+))?")


DFUTISSUE_CLASS_ORDER: List[Tuple[int, str]] = [
    (0, "Bg"),
    (1, "Fibrin"),
    (2, "Gran."),
    (3, "Callus"),
]
LUTSEG_CLASS_ORDER: List[Tuple[int, str]] = [
    (0, "Bg"),
    (1, "Epi"),
    (2, "Slough"),
    (3, "Gran."),
    (4, "Necr."),
    (5, "Other"),
]


@dataclass
class EvalBlock:
    miou: float
    ema: Optional[float]
    class_iou: Dict[int, float]
    class_ema: Dict[int, float]


@dataclass
class BestSelection:
    method: str
    dataset: str
    split: str
    metric_name: str
    metric_value: float
    seed: str
    log_path: Path
    run_id: str
    class_values: Dict[int, float]


def parse_log_blocks(log_path: Path) -> List[EvalBlock]:
    blocks: List[EvalBlock] = []
    current_iou: Dict[int, float] = {}
    current_ema: Dict[int, float] = {}

    with log_path.open("r", errors="ignore") as handle:
        for line in handle:
            class_match = CLASS_RE.search(line)
            if class_match:
                class_idx = int(class_match.group(1))
                class_iou = float(class_match.group(3))
                class_ema = (
                    float(class_match.group(4)) if class_match.group(4) is not None else None
                )
                current_iou[class_idx] = class_iou
                if class_ema is not None:
                    current_ema[class_idx] = class_ema
                continue

            mean_match = MEAN_RE.search(line)
            if mean_match:
                miou = float(mean_match.group(1))
                ema = float(mean_match.group(2)) if mean_match.group(2) is not None else None
                blocks.append(
                    EvalBlock(
                        miou=miou,
                        ema=ema,
                        class_iou=dict(current_iou),
                        class_ema=dict(current_ema),
                    )
                )
                current_iou.clear()
                current_ema.clear()

    return blocks


def select_best_block(
    blocks: List[EvalBlock], prefer_metric: str
) -> Optional[Tuple[str, float, Dict[int, float]]]:
    if not blocks:
        return None

    has_ema = any(block.ema is not None for block in blocks)
    use_ema = prefer_metric == "ema" and has_ema

    if use_ema:
        best_block = max(
            blocks, key=lambda block: float("-inf") if block.ema is None else block.ema
        )
        metric_name = "best_ema"
        metric_value = float(best_block.ema)
        class_values = best_block.class_ema if best_block.class_ema else best_block.class_iou
    else:
        best_block = max(blocks, key=lambda block: block.miou)
        metric_name = "best_miou"
        metric_value = best_block.miou
        class_values = best_block.class_iou

    return metric_name, metric_value, class_values


def split_seed_run_id(
    method_root: Path, log_path: Path, split_name: str
) -> Tuple[Optional[str], str, str]:
    rel = log_path.relative_to(method_root)
    parts = rel.parts
    split = split_name if split_name in parts else None
    seed = next((token for token in parts if token.startswith("seed")), "NA")
    return split, seed, rel.parent.as_posix()


def parse_seed_filter(value: str) -> Optional[set[str]]:
    seeds = {item.strip() for item in value.split(",") if item.strip()}
    if not seeds or "all" in seeds:
        return None
    return seeds


def collect_best_selection(
    method: str,
    dataset: str,
    method_root: Path,
    split_name: str,
    prefer_metric: str,
    exclude_seed_unknown: bool,
    allowed_seeds: Optional[set[str]],
) -> Optional[BestSelection]:
    if not method_root.exists():
        return None

    best_item: Optional[BestSelection] = None
    for log_path in method_root.rglob("out.log"):
        split, seed, run_id = split_seed_run_id(method_root, log_path, split_name)
        if split is None:
            continue
        if exclude_seed_unknown and seed == "seed_unknown":
            continue
        if allowed_seeds is not None and seed not in allowed_seeds:
            continue

        blocks = parse_log_blocks(log_path)
        selected = select_best_block(blocks, prefer_metric=prefer_metric)
        if selected is None:
            continue
        metric_name, metric_value, class_values = selected
        candidate = BestSelection(
            method=method,
            dataset=dataset,
            split=split_name,
            metric_name=metric_name,
            metric_value=metric_value,
            seed=seed,
            log_path=log_path,
            run_id=run_id,
            class_values=class_values,
        )
        if best_item is None or candidate.metric_value > best_item.metric_value:
            best_item = candidate

    return best_item


def fmt_class(value: Optional[float]) -> str:
    if value is None:
        return "xx.x"
    return f"{value:.1f}"


def fmt_miou(value: Optional[float]) -> str:
    if value is None:
        return "xx.xx"
    return f"{value:.2f}"


def extract_ordered_classes(
    class_values: Optional[Dict[int, float]], class_order: List[Tuple[int, str]]
) -> List[Optional[float]]:
    if class_values is None:
        return [None] * len(class_order)
    return [class_values.get(idx) for idx, _ in class_order]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract per-class IoU table values for UniMatch-V2 and TiSage."
    )
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to UniMatch-V2/exp.",
    )
    parser.add_argument(
        "--dfutissue-split",
        type=str,
        default="fixed",
        help="DFUTissue split for per-class table.",
    )
    parser.add_argument(
        "--lutseg-split",
        type=str,
        default="1_8",
        help="LUTSeg split for per-class table.",
    )
    parser.add_argument(
        "--prefer-metric",
        choices=["ema", "miou"],
        default="ema",
        help="Metric used to select the checkpoint block from each log.",
    )
    parser.add_argument(
        "--include-seed-unknown",
        action="store_true",
        help="Include seed_unknown logs (excluded by default).",
    )
    parser.add_argument(
        "--dfutissue-unimatch-seeds",
        type=str,
        default="all",
        help="Comma-separated seeds for DFUTissue UniMatch-V2 (default: all).",
    )
    parser.add_argument(
        "--lutseg-unimatch-seeds",
        type=str,
        default="seed0",
        help="Comma-separated seeds for LUTSeg UniMatch-V2 (default: seed0).",
    )
    parser.add_argument(
        "--dfutissue-tisage-seeds",
        type=str,
        default="all",
        help="Comma-separated seeds for DFUTissue TiSage (default: all).",
    )
    parser.add_argument(
        "--lutseg-tisage-seeds",
        type=str,
        default="seed0",
        help="Comma-separated seeds for LUTSeg TiSage (default: seed0).",
    )
    args = parser.parse_args()

    exp_root = args.exp_root.resolve()
    dfutissue_root = exp_root / "dfutissue"
    lutseg_root = exp_root / "lutseg"
    exclude_seed_unknown = not args.include_seed_unknown

    seed_filters = {
        ("UniMatch-V2", "dfutissue"): parse_seed_filter(args.dfutissue_unimatch_seeds),
        ("UniMatch-V2", "lutseg"): parse_seed_filter(args.lutseg_unimatch_seeds),
        ("TiSage (Ours)", "dfutissue"): parse_seed_filter(args.dfutissue_tisage_seeds),
        ("TiSage (Ours)", "lutseg"): parse_seed_filter(args.lutseg_tisage_seeds),
    }

    methods = [
        (
            "UniMatch-V2",
            dfutissue_root / "unimatch_v2/dinov2_base",
            lutseg_root / "unimatch_v2_baseline/dinov2_base",
        ),
        (
            "TiSage (Ours)",
            dfutissue_root / "unimatch_v2_medsiglip_multiscale_adaptive/dinov2_base",
            lutseg_root / "unimatch_v2_medsiglip_multiscale_adaptive/dinov2_base",
        ),
    ]

    print(f"exp_root: {exp_root}")
    print(f"prefer_metric: {args.prefer_metric}")
    print(f"dfutissue_split: {args.dfutissue_split}")
    print(f"lutseg_split: {args.lutseg_split}")
    print(f"exclude_seed_unknown: {exclude_seed_unknown}")
    print("")

    print(
        "Method\t"
        "DFU_Bg\tDFU_Fibrin\tDFU_Gran\tDFU_Callus\tDFU_mIoU\t"
        "LUT_Bg\tLUT_Epi\tLUT_Slough\tLUT_Gran\tLUT_Necr\tLUT_Other\tLUT_mIoU"
    )

    for method_name, dfu_method_root, lut_method_root in methods:
        dfu_best = collect_best_selection(
            method=method_name,
            dataset="dfutissue",
            method_root=dfu_method_root,
            split_name=args.dfutissue_split,
            prefer_metric=args.prefer_metric,
            exclude_seed_unknown=exclude_seed_unknown,
            allowed_seeds=seed_filters[(method_name, "dfutissue")],
        )
        lut_best = collect_best_selection(
            method=method_name,
            dataset="lutseg",
            method_root=lut_method_root,
            split_name=args.lutseg_split,
            prefer_metric=args.prefer_metric,
            exclude_seed_unknown=exclude_seed_unknown,
            allowed_seeds=seed_filters[(method_name, "lutseg")],
        )

        dfu_class_vals = extract_ordered_classes(
            None if dfu_best is None else dfu_best.class_values, DFUTISSUE_CLASS_ORDER
        )
        lut_class_vals = extract_ordered_classes(
            None if lut_best is None else lut_best.class_values, LUTSEG_CLASS_ORDER
        )

        row = [
            method_name,
            *[fmt_class(v) for v in dfu_class_vals],
            fmt_miou(None if dfu_best is None else dfu_best.metric_value),
            *[fmt_class(v) for v in lut_class_vals],
            fmt_miou(None if lut_best is None else lut_best.metric_value),
        ]
        print("\t".join(row))

        if dfu_best is not None:
            print(
                f"  source_dfutissue: metric={dfu_best.metric_name} seed={dfu_best.seed} "
                f"log={dfu_best.log_path} run_id={dfu_best.run_id}"
            )
        else:
            print("  source_dfutissue: NA")
        if lut_best is not None:
            print(
                f"  source_lutseg: metric={lut_best.metric_name} seed={lut_best.seed} "
                f"log={lut_best.log_path} run_id={lut_best.run_id}"
            )
        else:
            print("  source_lutseg: NA")

        latex_row = (
            f"{method_name} & "
            + " & ".join([fmt_class(v) for v in dfu_class_vals])
            + f" & {fmt_miou(None if dfu_best is None else dfu_best.metric_value)} & "
            + " & ".join([fmt_class(v) for v in lut_class_vals])
            + f" & {fmt_miou(None if lut_best is None else lut_best.metric_value)} \\\\"
        )
        print(f"  latex_row: {latex_row}")
        print("")


if __name__ == "__main__":
    main()
