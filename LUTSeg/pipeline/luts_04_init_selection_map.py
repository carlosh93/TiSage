#!/usr/bin/env python3
"""
Initialize per-image doctor selection mapping.

Useful bootstrap before form voting:
- Prefer one doctor globally (e.g., user_9) when available.
- Fallback to first available annotator for images where preferred doctor is absent.

Output format:
{
  "selections": {
    "Patient_1/P1_T1.jpeg": "user_9",
    ...
  }
}
"""
from __future__ import annotations

import argparse

from common import parse_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize selected_doctor_by_image mapping.")
    parser.add_argument(
        "--groups-json",
        default="data/LUTS/Annotations/processed/image_groups.json",
        help="Image groups JSON path (step 3 output).",
    )
    parser.add_argument(
        "--preferred-doctor",
        default="user_9",
        help="Preferred doctor_id to use when present in an image group.",
    )
    parser.add_argument(
        "--golden-only",
        action="store_true",
        help="Initialize mapping only for golden-set images.",
    )
    parser.add_argument(
        "--output-json",
        default="data/LUTS/Annotations/processed/selected_doctor_by_image.json",
        help="Output mapping JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = parse_json(args.groups_json)
    if not isinstance(groups, list) or not groups:
        raise SystemExit("Image groups JSON is empty or invalid.")

    selections: dict[str, str] = {}
    n_preferred = 0
    n_fallback = 0

    for group in groups:
        if args.golden_only and not bool(group.get("is_golden_patient")):
            continue
        image_key = str(group.get("image_key") or "")
        anns = group.get("annotations", []) or []
        doctor_ids = sorted({str(a.get("doctor_id")) for a in anns if a.get("doctor_id")})
        if not image_key or not doctor_ids:
            continue
        if args.preferred_doctor in doctor_ids:
            selections[image_key] = args.preferred_doctor
            n_preferred += 1
        else:
            selections[image_key] = doctor_ids[0]
            n_fallback += 1

    if not selections:
        raise SystemExit("No selection entries generated.")

    output = {"selections": selections}
    write_json(args.output_json, output)
    print(f"Wrote mapping: {args.output_json}")
    print(f"Entries: {len(selections)}")
    print(f"Preferred doctor ({args.preferred_doctor}): {n_preferred}")
    print(f"Fallback first available: {n_fallback}")


if __name__ == "__main__":
    main()
