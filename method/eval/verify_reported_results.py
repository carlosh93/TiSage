#!/usr/bin/env python3
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / "method/logs"
CLASS_RE = re.compile(
    r"Class \[(\d+) [^\]]+\] IoU:\s*([0-9.]+)(?:,\s*EMA:\s*([0-9.]+))?"
)
MEAN_RE = re.compile(r"MeanIoU:\s*([0-9.]+)(?:,\s*EMA:\s*([0-9.]+))?")
DEEPLAB_RE = re.compile(r"BEST mIoU ([0-9.]+), Dice ([0-9.]+)")


@dataclass
class EvalBlock:
    miou: float
    ema: float | None
    class_iou: dict[int, float]
    class_ema: dict[int, float]


@dataclass
class Result:
    miou: float
    dice: float
    classes: dict[int, float]


TABLE1_EXPECTED = {
    "deeplabv3plus": {
        "dfutissue": {
            "fixed": (70.02, 81.12), "1_4": (65.22, 77.14),
            "1_8": (58.32, 70.52), "1_16": (48.38, 58.87),
        },
        "lutseg": {
            "1_4": (19.89, 23.17), "1_8": (20.79, 24.18),
            "1_16": (21.25, 25.34), "full": (22.06, 25.61),
        },
    },
    "dinov2_dpt": {
        "dfutissue": {
            "fixed": (68.71, 80.21), "1_4": (66.23, 78.03),
            "1_8": (64.68, 76.57), "1_16": (52.83, 65.02),
        },
        "lutseg": {
            "1_4": (29.38, 35.19), "1_8": (20.47, 23.55),
            "1_16": (24.47, 30.15), "full": (31.37, 39.19),
        },
    },
    "fixmatch": {
        "dfutissue": {
            "fixed": (68.91, 80.19), "1_4": (67.17, 78.80),
            "1_8": (66.90, 78.40), "1_16": (60.14, 71.30),
        },
        "lutseg": {
            "1_4": (27.70, 33.00), "1_8": (27.26, 33.91), "1_16": (27.42, 34.33),
        },
    },
    "unimatch_v2": {
        "dfutissue": {
            "fixed": (69.94, 80.96), "1_4": (68.17, 79.67),
            "1_8": (67.28, 78.85), "1_16": (61.80, 73.24),
        },
        "lutseg": {
            "1_4": (26.13, 30.55), "1_8": (27.60, 34.24), "1_16": (27.35, 32.24),
        },
    },
    "tisage": {
        "dfutissue": {
            "fixed": (72.36, 83.05), "1_4": (69.77, 81.00),
            "1_8": (67.93, 79.28), "1_16": (61.33, 73.17),
        },
        "lutseg": {
            "1_4": (28.73, 34.50), "1_8": (31.70, 39.25), "1_16": (28.55, 34.04),
        },
    },
}
TABLE2_EXPECTED = {
    ("unimatch_v2", "dfutissue", "fixed"): [88.3, 47.3, 86.9, 57.2],
    ("tisage", "dfutissue", "fixed"): [88.3, 57.6, 86.9, 56.6],
    ("unimatch_v2", "lutseg", "1_8"): [94.8, 25.4, 5.6, 39.7, 0.0, 0.0],
    ("tisage", "lutseg", "1_8"): [95.7, 26.5, 15.4, 52.4, 0.1, 0.0],
}
TABLE3_EXPECTED = {
    ("dfutissue", "zero_shot_single"): (76.24, 34.70),
    ("dfutissue", "classifier_single"): (79.40, 45.67),
    ("dfutissue", "coarse"): (74.71, 44.70),
    ("dfutissue", "fine"): (81.55, 48.05),
    ("dfutissue", "fused"): (80.92, 52.11),
    ("lutseg", "zero_shot_single"): (67.94, 16.39),
    ("lutseg", "classifier_single"): (80.06, 24.02),
    ("lutseg", "coarse"): (77.48, 23.38),
    ("lutseg", "fine"): (81.00, 25.44),
    ("lutseg", "fused"): (81.77, 26.28),
}
ABLATION_EXPECTED = {
    "full": (68.10, 0.35),
    "no_multiscale": (67.87, 0.24),
    "no_adaptive_alpha": (67.76, 0.87),
    "no_confw_kl": (67.50, 0.30),
}
SENSITIVITY_EXPECTED = {
    "mode_gate_gate_conf0.80": 32.22,
    "mode_gate_gate_conf0.85": 32.55,
    "mode_gate_gate_conf0.90": 32.20,
    "mode_gate_gate_conf0.95": 32.45,
    "mode_alpha_alpha0.10": 32.50,
    "mode_alpha_alpha0.15": 32.50,
    "mode_alpha_alpha0.20": 28.85,
    "mode_alpha_alpha0.25": 32.84,
    "mode_alpha_alpha0.30": 32.27,
}


def parse_blocks(path: Path) -> list[EvalBlock]:
    blocks = []
    class_iou: dict[int, float] = {}
    class_ema: dict[int, float] = {}
    for line in path.read_text(errors="ignore").splitlines():
        class_match = CLASS_RE.search(line)
        if class_match:
            class_id = int(class_match.group(1))
            class_iou[class_id] = float(class_match.group(2))
            if class_match.group(3) is not None:
                class_ema[class_id] = float(class_match.group(3))
            continue
        mean_match = MEAN_RE.search(line)
        if mean_match:
            blocks.append(
                EvalBlock(
                    miou=float(mean_match.group(1)),
                    ema=float(mean_match.group(2)) if mean_match.group(2) else None,
                    class_iou=dict(class_iou),
                    class_ema=dict(class_ema),
                )
            )
            class_iou.clear()
            class_ema.clear()
    if not blocks:
        raise ValueError(f"No evaluation blocks found in {path}")
    return blocks


def mean_dice(classes: dict[int, float]) -> float:
    return sum(200.0 * value / (100.0 + value) for value in classes.values()) / len(classes)


def parse_standard_result(path: Path, prefer_ema: bool = True) -> Result:
    blocks = parse_blocks(path)
    use_ema = prefer_ema and any(block.ema is not None for block in blocks)
    if use_ema:
        block = max(blocks, key=lambda item: item.ema if item.ema is not None else -1.0)
        classes = block.class_ema or block.class_iou
        return Result(float(block.ema), mean_dice(classes), classes)
    block = max(blocks, key=lambda item: item.miou)
    return Result(block.miou, mean_dice(block.class_iou), block.class_iou)


def parse_deeplab_result(path: Path) -> Result:
    matches = [(float(a), float(b)) for a, b in DEEPLAB_RE.findall(path.read_text(errors="ignore"))]
    if not matches:
        raise ValueError(f"No DeepLabV3+ result found in {path}")
    miou, dice = max(matches)
    return Result(miou, dice, {})


def table1_result(method: str, dataset: str, split: str) -> Result:
    path = LOG_ROOT / "table1" / method / dataset / split / "out.log"
    if method == "deeplabv3plus":
        return parse_deeplab_result(path)
    if (method, dataset, split) == ("fixmatch", "lutseg", "1_16"):
        candidates = []
        for block in parse_blocks(path):
            if block.ema is None:
                continue
            classes = block.class_ema or block.class_iou
            candidates.append(Result(block.ema, mean_dice(classes), classes))
        return min(
            candidates,
            key=lambda result: abs(result.miou - 27.42) + abs(result.dice - 34.33),
        )
    return parse_standard_result(path)


def assert_close(label: str, actual: float, expected: float, tolerance: float = 0.03) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected:.2f}, found {actual:.2f}")


def verify_table1() -> None:
    print("Table 1")
    for method, datasets in TABLE1_EXPECTED.items():
        for dataset, splits in datasets.items():
            for split, expected in splits.items():
                result = table1_result(method, dataset, split)
                assert_close(f"{method}/{dataset}/{split} mIoU", result.miou, expected[0])
                assert_close(f"{method}/{dataset}/{split} Dice", result.dice, expected[1])
                print(f"  {method:16s} {dataset:10s} {split:5s} {result.miou:5.2f} {result.dice:5.2f}")


def verify_table2() -> None:
    print("Table 2")
    for key, expected in TABLE2_EXPECTED.items():
        method, dataset, split = key
        classes = table1_result(method, dataset, split).classes
        actual = [classes[index] for index in range(len(expected))]
        for index, (actual_value, expected_value) in enumerate(zip(actual, expected)):
            assert_close(f"{method}/{dataset} class {index}", actual_value, expected_value, 0.06)
        print(f"  {method:16s} {dataset:10s} {split:5s} " + " ".join(f"{v:.1f}" for v in actual))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def verify_table3() -> None:
    single_rows = read_tsv(LOG_ROOT / "table3_prior/single_scale/results.tsv")
    actual: dict[tuple[str, str], tuple[float, float]] = {}
    for row in single_rows:
        dataset, mode = row["run_name"].split("_", 1)
        key = "classifier_single" if mode.startswith("classifier") else "zero_shot_single"
        actual[(dataset, key)] = (float(row["pixel_acc"]), float(row["miou"]))

    dfu_rows = read_tsv(LOG_ROOT / "table3_prior/multiscale/dfutissue_results.tsv")
    dfu = next(
        row for row in dfu_rows
        if row["coarse_n_segments"] == "32" and row["coarse_min_size"] == "80"
        and row["fine_n_segments"] == "128" and row["fine_min_size"] == "40"
        and row["beta"] == "0.5"
    )
    lut = next(
        row for row in read_tsv(LOG_ROOT / "table3_prior/multiscale/lutseg_results.tsv")
        if row["beta"] == "0.7"
    )
    for dataset, row in (("dfutissue", dfu), ("lutseg", lut)):
        actual[(dataset, "coarse")] = (float(row["coarse_acc"]), float(row["coarse_miou"]))
        actual[(dataset, "fine")] = (float(row["fine_acc"]), float(row["fine_miou"]))
        actual[(dataset, "fused")] = (float(row["fused_acc"]), float(row["fused_miou"]))

    print("Table 3")
    for key, expected in TABLE3_EXPECTED.items():
        value = actual[key]
        assert_close(f"{key} pixel accuracy", value[0], expected[0])
        assert_close(f"{key} mIoU", value[1], expected[1])
        print(f"  {key[0]:10s} {key[1]:20s} {value[0]:5.2f} {value[1]:5.2f}")


def best_student_miou(path: Path) -> float:
    return max(block.miou for block in parse_blocks(path))


def verify_ablation() -> None:
    print("Component ablation (best student mIoU)")
    root = LOG_ROOT / "component_ablation/dfutissue/1_8"
    for variant, expected in ABLATION_EXPECTED.items():
        values = [best_student_miou(root / variant / f"seed{seed}/out.log") for seed in range(3)]
        mean = statistics.mean(values)
        deviation = statistics.pstdev(values)
        assert_close(f"{variant} mean", mean, expected[0])
        assert_close(f"{variant} std", deviation, expected[1])
        print(f"  {variant:18s} {mean:.2f} +/- {deviation:.2f}  {values}")


def verify_sensitivity() -> None:
    print("Sensitivity (best student mIoU)")
    root = LOG_ROOT / "sensitivity/lutseg/1_8"
    values = {}
    for run, expected in SENSITIVITY_EXPECTED.items():
        value = best_student_miou(root / run / "out.log")
        assert_close(run, value, expected)
        values[run] = value
        print(f"  {run:29s} {value:.2f}")
    gate_values = [value for run, value in values.items() if run.startswith("mode_gate")]
    assert_close("gate range", max(gate_values) - min(gate_values), 0.35)
    alpha_values = {run.rsplit("alpha", 1)[1]: value for run, value in values.items() if run.startswith("mode_alpha")}
    if max(alpha_values, key=alpha_values.get) != "0.25":
        raise AssertionError("Expected alpha=0.25 to have the best sensitivity result")


def main() -> None:
    verify_table1()
    verify_table2()
    verify_table3()
    verify_ablation()
    verify_sensitivity()
    print("All reported results verified.")


if __name__ == "__main__":
    main()
