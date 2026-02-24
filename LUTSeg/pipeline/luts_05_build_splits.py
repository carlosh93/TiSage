#!/usr/bin/env python3
"""
Step 5: Build patient-level train/val splits for LUTS (no test split).

Outputs:
- data/LUTS/Annotations/processed/splits.json
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict

from common import parse_csv_arg, parse_json, write_json


DEFAULT_GOLDEN = (
    "Patient_1,Patient_5,Patient_6,Patient_11,Patient_13,"
    "Patient_20,Patient_28,Patient_33,Patient_39"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build LUTS patient-level splits.")
    parser.add_argument(
        "--consensus-manifest",
        default="data/LUTS/Annotations/processed/consensus_manifest.json",
        help="Step-4 manifest JSON path (selected-doctor or consensus).",
    )
    parser.add_argument(
        "--output-json",
        default="data/LUTS/Annotations/processed/splits.json",
        help="Output split JSON path.",
    )
    parser.add_argument(
        "--golden-patients",
        default=DEFAULT_GOLDEN,
        help="Comma-separated golden patient IDs.",
    )
    parser.add_argument(
        "--val-patients",
        default="",
        help="Optional comma-separated explicit val patients.",
    )
    parser.add_argument(
        "--val-ratio-golden",
        type=float,
        default=0.5,
        help="If val not explicit: fraction of golden patients used for val (rest train).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for auto split selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = parse_json(args.consensus_manifest)
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("Consensus manifest is empty or invalid.")

    patients_in_data = sorted(
        {m.get("patient_id") for m in manifest if isinstance(m.get("patient_id"), str)}
    )
    golden = set(parse_csv_arg(args.golden_patients))
    val_patients = set(parse_csv_arg(args.val_patients))

    if not val_patients:
        golden_present = sorted(golden.intersection(patients_in_data))
        if len(golden_present) < 1:
            raise SystemExit("No golden patients present for val split.")
        rnd = random.Random(args.seed)
        rnd.shuffle(golden_present)
        # Use round-half-up (instead of Python's bankers rounding) so 9 * 0.5 -> 5.
        n_val = max(1, int(len(golden_present) * args.val_ratio_golden + 0.5))
        n_val = min(n_val, len(golden_present))
        val_patients = set(golden_present[:n_val])

    split_by_patient: dict[str, str] = {}
    for p in patients_in_data:
        if p in val_patients:
            split_by_patient[p] = "val"
        else:
            split_by_patient[p] = "train"

    by_split = defaultdict(list)
    for item in manifest:
        patient = item.get("patient_id")
        if not patient:
            continue
        split = split_by_patient.get(patient, "train")
        by_split[split].append(
            {
                "image_name": item.get("image_name"),
                "image_key": item.get("image_key"),
                "patient_id": patient,
                "source_image_path": item.get("source_image_path"),
                "mask_path": item.get("mask_path"),
                "wound_mask_path": item.get("wound_mask_path"),
            }
        )

    output = {
        "patients": {
            "train": sorted([p for p, s in split_by_patient.items() if s == "train"]),
            "val": sorted([p for p, s in split_by_patient.items() if s == "val"]),
        },
        "counts": {
            "train": len(by_split["train"]),
            "val": len(by_split["val"]),
            "total": len(manifest),
        },
        "samples": {
            "train": by_split["train"],
            "val": by_split["val"],
        },
    }

    write_json(args.output_json, output)
    print(f"Wrote splits: {args.output_json}")
    print(f"Counts: {output['counts']}")
    print(f"Val patients: {output['patients']['val']}")


if __name__ == "__main__":
    main()
