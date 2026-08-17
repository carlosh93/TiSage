#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
EXPECTED_SPLIT_SIZES = {"train.txt": 111, "val.txt": 30}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the lightweight TiSage and LUTSeg release smoke test"
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/LUTSeg")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download or update the public LUTSeg snapshot before testing",
    )
    return parser.parse_args()


def download_dataset(data_root: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="ksanchez84/LUTSeg",
        repo_type="dataset",
        local_dir=data_root,
    )


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number} must contain an image-mask pair")
        pairs.append((fields[0], fields[1]))
    return pairs


def check_dataset(data_root: Path) -> None:
    required = (
        "Images",
        "Masks",
        "Masks_RGB",
        "Wound_Masks",
        "metadata.jsonl",
        "checksums.sha256",
    )
    missing = [name for name in required if not (data_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing LUTSeg release entries: {', '.join(missing)}")

    splits = {}
    for filename, expected_size in EXPECTED_SPLIT_SIZES.items():
        pairs = read_pairs(data_root / filename)
        if len(pairs) != expected_size:
            raise ValueError(f"{filename}: expected {expected_size} rows, found {len(pairs)}")
        splits[filename] = pairs

    all_pairs = splits["train.txt"] + splits["val.txt"]
    if len(set(all_pairs)) != 141:
        raise ValueError("train and validation pairs must be unique and disjoint")
    for image_relative, mask_relative in all_pairs:
        paths = (
            data_root / image_relative,
            data_root / mask_relative,
            data_root / mask_relative.replace("Masks/", "Masks_RGB/", 1),
            data_root / mask_relative.replace("Masks/", "Wound_Masks/", 1),
        )
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError(f"incomplete sample: {image_relative}")

    metadata = [json.loads(line) for line in (data_root / "metadata.jsonl").read_text().splitlines()]
    if len(metadata) != 141:
        raise ValueError(f"metadata.jsonl: expected 141 rows, found {len(metadata)}")

    for image_relative, mask_relative in (splits["train.txt"][0], splits["val.txt"][0]):
        with Image.open(data_root / image_relative) as image, Image.open(
            data_root / mask_relative
        ) as mask:
            if image.size != mask.size:
                raise ValueError(f"image-mask size mismatch: {image_relative}")
            labels = set(mask.getdata())
            if not labels <= set(range(6)):
                raise ValueError(f"unexpected tissue labels in {mask_relative}: {sorted(labels)}")

    train_patients = {image.split("/", 2)[1] for image, _ in splits["train.txt"]}
    val_patients = {image.split("/", 2)[1] for image, _ in splits["val.txt"]}
    if train_patients & val_patients or len(train_patients | val_patients) != 39:
        raise ValueError("patient-level train/validation partition is invalid")
    print("PASS dataset: 141 samples, 39 patients, patient-disjoint 111/30 split")


def check_runtime_loader(data_root: Path) -> None:
    from dataset.semi import SemiDataset

    dataset = SemiDataset(
        "lutseg",
        str(data_root),
        "train_l",
        size=518,
        id_path=str(ROOT / "splits/lutseg/1_8/labeled.txt"),
    )
    image, mask = dataset[0]
    if len(dataset) != 13 or tuple(image.shape) != (3, 518, 518) or tuple(mask.shape) != (518, 518):
        raise ValueError("TiSage LUTSeg loader returned unexpected shapes or split size")
    print("PASS loader: LUTSeg 1/8 sample loaded through the TiSage runtime")


def check_command(label: str, command: list[str]) -> None:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    if result.returncode:
        output = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"{label} failed:\n{output}")
    print(f"PASS {label}")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if args.download:
        download_dataset(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"{data_root} does not exist; rerun with --download")

    check_dataset(data_root)
    check_runtime_loader(data_root)
    check_command(
        "training entrypoint",
        [sys.executable, "method/src/tisage/train.py", "--help"],
    )
    check_command(
        "committed result evidence",
        [sys.executable, "method/eval/verify_reported_results.py"],
    )
    check_command(
        "Table 1 dry run",
        [
            sys.executable,
            "method/scripts/run_table1.py",
            "--method",
            "tisage",
            "--dataset",
            "lutseg",
            "--split",
            "1_8",
            "--dry-run",
        ],
    )
    print("TiSage release smoke test: PASS (no GPU training was launched)")


if __name__ == "__main__":
    main()
