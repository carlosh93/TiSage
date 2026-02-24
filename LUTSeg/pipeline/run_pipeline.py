#!/usr/bin/env python3
"""
Run the full LUTS pipeline (steps 1-7).

Usage:
  # From repository root: selection JSON is auto-created before step 4 if missing (from image_groups + default-doctor).
  venv/bin/python data/LUTS/pipeline/run_pipeline.py --default-doctor user_9

  # Use explicit per-image mapping (e.g. from votes):
  venv/bin/python data/LUTS/pipeline/run_pipeline.py \
      --selection-json data/LUTS/Annotations/processed/selected_doctor_by_image.json

  # Fail if selection JSON is missing (e.g. you expect to have run votes first):
  venv/bin/python data/LUTS/pipeline/run_pipeline.py --no-auto-init-selection

Requires:
- Raw Label Studio exports in data/LUTS/Annotations/raw/*.json
- Source images under data/Dataset_evolution_wounds_VR/
- Python with opencv-python (cv2) and numpy for steps 2, 4, 7 (use project venv if needed)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LUTS pipeline (normalize → rasterize → group → selected-mask step → splits → export → QC)."
    )
    parser.add_argument(
        "--selection-json",
        default="data/LUTS/Annotations/processed/selected_doctor_by_image.json",
        help=(
            "JSON mapping image_key -> doctor_id for step 4 selection. "
            "If missing entries, --default-doctor is used when available."
        ),
    )
    parser.add_argument(
        "--default-doctor",
        default="user_9",
        help="Fallback doctor_id for step 4 when image_key is not in selection mapping.",
    )
    parser.add_argument(
        "--strict-selection",
        action="store_true",
        help="Fail step 4 if neither mapped doctor nor default doctor is available for an image.",
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="Repository root directory. Default: inferred from this script path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Allow overwriting train/val.txt in step 6 (default: True).",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Do not overwrite existing split files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only, do not run.",
    )
    parser.add_argument(
        "--auto-init-selection",
        action="store_true",
        default=True,
        help="If selection JSON is missing before step 4, run luts_04_init_selection_map.py to create it from image_groups (default: True).",
    )
    parser.add_argument(
        "--no-auto-init-selection",
        dest="auto_init_selection",
        action="store_false",
        help="Do not auto-create selection JSON; fail if missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Project root (so that data/LUTS/... and data/Dataset_evolution_wounds_VR resolve):
    # run_pipeline.py lives in <project>/data/LUTS/pipeline → parents[2] = project root
    script_dir = Path(__file__).resolve().parent
    repo_root = Path(args.repo_root).resolve() if args.repo_root else script_dir.parents[2]

    pipeline_dir = script_dir
    steps = [
        ("1_normalize", [sys.executable, str(pipeline_dir / "luts_01_normalize_exports.py")]),
        ("2_rasterize", [sys.executable, str(pipeline_dir / "luts_02_rasterize_masks.py")]),
        ("3_groups", [sys.executable, str(pipeline_dir / "luts_03_build_image_groups.py")]),
        (
            "4_select",
            [
                sys.executable,
                str(pipeline_dir / "luts_04_select_masks.py"),
                "--selection-json",
                str(args.selection_json),
                "--default-doctor",
                str(args.default_doctor),
            ]
            + (["--strict-selection"] if args.strict_selection else []),
        ),
        ("5_splits", [sys.executable, str(pipeline_dir / "luts_05_build_splits.py")]),
        (
            "6_export",
            [sys.executable, str(pipeline_dir / "luts_06_export_dataset_layout.py")]
            + (["--overwrite"] if args.overwrite else []),
        ),
        ("7_qc", [sys.executable, str(pipeline_dir / "luts_07_qc_report.py")]),
    ]

    selection_path = repo_root / args.selection_json if not Path(args.selection_json).is_absolute() else Path(args.selection_json)

    if args.dry_run:
        print(f"Repo root: {repo_root}")
        for name, cmd in steps:
            print(" ", name, " ".join(cmd))
        if args.auto_init_selection and not selection_path.is_absolute():
            print("  (selection JSON will be auto-created before step 4 if missing)")
        return

    for i, (name, cmd) in enumerate(steps):
        # Before step 4: create selection JSON if missing and auto-init is enabled
        if name == "4_select" and args.auto_init_selection and not selection_path.exists():
            print(f"\n--- Auto-init selection (missing {selection_path}) ---")
            init_cmd = [
                sys.executable,
                str(pipeline_dir / "luts_04_init_selection_map.py"),
                "--groups-json",
                str(repo_root / "data/LUTS/Annotations/processed/image_groups.json"),
                "--preferred-doctor",
                str(args.default_doctor),
                "--output-json",
                str(selection_path),
            ]
            r = subprocess.run(init_cmd, cwd=repo_root)
            if r.returncode != 0:
                print(f"Auto-init selection failed with exit code {r.returncode}", file=sys.stderr)
                sys.exit(r.returncode)
        print(f"\n--- Step {name} ---")
        r = subprocess.run(cmd, cwd=repo_root)
        if r.returncode != 0:
            print(f"Step {name} failed with exit code {r.returncode}", file=sys.stderr)
            sys.exit(r.returncode)

    print("\n--- Pipeline finished successfully ---")


if __name__ == "__main__":
    main()
