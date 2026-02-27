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


@dataclass
class EvalBlock:
    miou: float
    ema: Optional[float]
    class_iou: Dict[int, float]
    class_ema: Dict[int, float]


@dataclass
class BestRun:
    metric_name: str
    metric_value: float
    mdice: float
    split: str
    seed: str
    log_path: Path
    run_id: str


@dataclass
class MethodRoots:
    name: str
    dfutissue_subdir: str
    lutseg_subdir: str


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


def mean_dice_from_ious(iou_by_class: Dict[int, float]) -> float:
    if not iou_by_class:
        return float("nan")
    dices = [(2.0 * iou) / (100.0 + iou) * 100.0 for iou in iou_by_class.values()]
    return sum(dices) / len(dices)


def best_metric_from_log(
    log_path: Path, prefer_metric: str
) -> Optional[Tuple[str, float, float]]:
    blocks = parse_log_blocks(log_path)
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
        class_for_dice = best_block.class_ema if best_block.class_ema else best_block.class_iou
    else:
        best_block = max(blocks, key=lambda block: block.miou)
        metric_name = "best_miou"
        metric_value = best_block.miou
        class_for_dice = best_block.class_iou

    mdice = mean_dice_from_ious(class_for_dice)
    return metric_name, metric_value, mdice


def extract_split_seed_run_id(
    method_root: Path, log_path: Path, valid_splits: List[str]
) -> Tuple[Optional[str], str, str]:
    rel = log_path.relative_to(method_root)
    parts = rel.parts

    split = next((token for token in parts if token in valid_splits), None)
    seed = next((token for token in parts if token.startswith("seed")), "NA")

    run_parent = rel.parent
    run_id = run_parent.as_posix()
    return split, seed, run_id


def collect_best_per_split(
    method_root: Path,
    splits: List[str],
    prefer_metric: str,
    exclude_seed_unknown: bool,
    allowed_seeds: Optional[set[str]] = None,
) -> Dict[str, Optional[BestRun]]:
    out: Dict[str, Optional[BestRun]] = {split: None for split in splits}
    if not method_root.exists():
        return out

    for log_path in method_root.rglob("out.log"):
        split, seed, run_id = extract_split_seed_run_id(method_root, log_path, splits)
        if split is None:
            continue
        if exclude_seed_unknown and seed == "seed_unknown":
            continue
        if allowed_seeds is not None and seed not in allowed_seeds:
            continue

        best = best_metric_from_log(log_path, prefer_metric=prefer_metric)
        if best is None:
            continue
        metric_name, metric_value, mdice = best
        candidate = BestRun(
            metric_name=metric_name,
            metric_value=metric_value,
            mdice=mdice,
            split=split,
            seed=seed,
            log_path=log_path,
            run_id=run_id,
        )
        prev = out[split]
        if prev is None or candidate.metric_value > prev.metric_value:
            out[split] = candidate

    return out


def fmt(v: float) -> str:
    return "nan" if v != v else f"{v:.2f}"


def render_split_block(
    title: str, best_by_split: Dict[str, Optional[BestRun]], splits: List[str]
) -> List[str]:
    lines = [title, "split\tmIoU\tDice\tmetric\tseed\tlog_path\trun_id"]
    for split in splits:
        item = best_by_split.get(split)
        if item is None:
            lines.append(f"{split}\tNA\tNA\tNA\tNA\tNA\tNA")
            continue
        lines.append(
            f"{split}\t{fmt(item.metric_value)}\t{fmt(item.mdice)}\t"
            f"{item.metric_name}\t{item.seed}\t{item.log_path}\t{item.run_id}"
        )
    return lines


def row_values(best_by_split: Dict[str, Optional[BestRun]], splits: List[str]) -> List[str]:
    values: List[str] = []
    for split in splits:
        item = best_by_split.get(split)
        if item is None:
            values.extend(["xx.xx", "xx.xx"])
        else:
            values.extend([fmt(item.metric_value), fmt(item.mdice)])
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract UniMatch-V2 (baseline) and TiSage (ours) table values "
            "for DFUTissue and LUTSeg."
        )
    )
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Path to UniMatch-V2/exp.",
    )
    parser.add_argument(
        "--dfutissue-splits",
        type=str,
        default="fixed,1_4,1_8,1_16",
        help="Comma-separated split order for DFUTissue.",
    )
    parser.add_argument(
        "--lutseg-splits",
        type=str,
        default="1_4,1_8,1_16",
        help="Comma-separated split order for LUTSeg.",
    )
    parser.add_argument(
        "--prefer-metric",
        choices=["ema", "miou"],
        default="ema",
        help="Metric used to pick the best checkpoint from each log.",
    )
    parser.add_argument(
        "--include-seed-unknown",
        action="store_true",
        help="Include logs under seed_unknown (excluded by default).",
    )
    parser.add_argument(
        "--lutseg-unimatch-seeds",
        type=str,
        default="seed0",
        help=(
            "Comma-separated seeds to allow for LUTSeg UniMatch-V2 (default: seed0). "
            "Use 'all' to disable filtering."
        ),
    )
    args = parser.parse_args()

    exp_root = args.exp_root.resolve()
    dfu_root = exp_root / "dfutissue"
    lut_root = exp_root / "lutseg"
    dfu_splits = [x.strip() for x in args.dfutissue_splits.split(",") if x.strip()]
    lut_splits = [x.strip() for x in args.lutseg_splits.split(",") if x.strip()]
    exclude_seed_unknown = not args.include_seed_unknown
    lutseg_unimatch_seeds = {
        x.strip() for x in args.lutseg_unimatch_seeds.split(",") if x.strip()
    }
    if "all" in lutseg_unimatch_seeds:
        lutseg_unimatch_seeds = set()

    methods = [
        MethodRoots(
            name="UniMatch-V2",
            dfutissue_subdir="unimatch_v2/dinov2_base",
            lutseg_subdir="unimatch_v2_baseline/dinov2_base",
        ),
        MethodRoots(
            name="TiSage (Ours)",
            dfutissue_subdir="unimatch_v2_medsiglip_multiscale_adaptive/dinov2_base",
            lutseg_subdir="unimatch_v2_medsiglip_multiscale_adaptive/dinov2_base",
        ),
    ]

    print(f"exp_root: {exp_root}")
    print(f"dfutissue_root: {dfu_root}")
    print(f"lutseg_root: {lut_root}")
    print(f"prefer_metric: {args.prefer_metric}")
    print(f"exclude_seed_unknown: {exclude_seed_unknown}")
    print(
        "lutseg_unimatch_seed_filter: "
        + ("all" if not lutseg_unimatch_seeds else ",".join(sorted(lutseg_unimatch_seeds)))
    )
    print("")

    for method in methods:
        dfu_method_root = dfu_root / method.dfutissue_subdir
        lut_method_root = lut_root / method.lutseg_subdir

        dfu_best = collect_best_per_split(
            method_root=dfu_method_root,
            splits=dfu_splits,
            prefer_metric=args.prefer_metric,
            exclude_seed_unknown=exclude_seed_unknown,
        )
        lut_allowed_seeds: Optional[set[str]] = None
        if method.name == "UniMatch-V2" and lutseg_unimatch_seeds:
            lut_allowed_seeds = lutseg_unimatch_seeds
        lut_best = collect_best_per_split(
            method_root=lut_method_root,
            splits=lut_splits,
            prefer_metric=args.prefer_metric,
            exclude_seed_unknown=exclude_seed_unknown,
            allowed_seeds=lut_allowed_seeds,
        )

        print(f"{method.name}")
        print(f"dfutissue_method_root: {dfu_method_root}")
        for line in render_split_block("DFUTissue", dfu_best, dfu_splits):
            print(line)
        print(f"lutseg_method_root: {lut_method_root}")
        for line in render_split_block("LUTSeg", lut_best, lut_splits):
            print(line)

        dfu_values = row_values(dfu_best, dfu_splits)
        lut_values = row_values(lut_best, lut_splits)
        print("dfutissue_row_values:", " | ".join(dfu_values))
        print("lutseg_row_values:", " | ".join(lut_values))
        print("full_table_row_values:", " | ".join(dfu_values + lut_values))
        print("")


if __name__ == "__main__":
    main()
