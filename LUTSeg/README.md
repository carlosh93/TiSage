# LUTSeg Reproducibility

This folder contains the reproducibility assets for LUTSeg dataset construction from raw Label Studio exports, plus the expert-voting workflow used for golden-set selection.

## Structure

- `pipeline/`: preprocessing and review scripts.
- `annotations/raw/`: raw Label Studio exports and wound-outline fix files.
- `examples/`: A small, non-identifying subset and form-review artifacts are included for reproducibility. The full dataset is not released in this repository during double-blind review because some images may contain non-anonymized content. Upon acceptance, the full dataset will be publicly available.

All annotator references in this repository use pseudonymous IDs (for example `user_7`).

## What This Reproduces

1. Normalize and rasterize raw polygon annotations.
2. Build grouped image-level annotations across annotators.
3. Generate anonymized form composites for golden-set voting.
4. Aggregate votes and produce `selected_doctor_by_image.json`.
5. Select one final mask per image and export train/val dataset layout.

## Core Pipeline Scripts

Expected main scripts in `pipeline/`:

- `luts_01_normalize_exports.py`
- `luts_02_rasterize_masks.py`
- `luts_03_build_image_groups.py`
- `luts_04_init_selection_map.py`
- `luts_04_select_masks.py`
- `luts_05_build_splits.py`
- `luts_06_export_dataset_layout.py`
- `luts_07_qc_report.py`
- `luts_generate_form_images.py`
- `luts_form_responses_to_votes.py`
- `luts_votes_to_selection_map.py`

## Expert Voting Workflow

Input form assets:

- `examples/Form/public/images/`
- `examples/Form/public/form_index.csv`
- `examples/Form/public/votes_template.csv`

Post-vote artifacts:

- `examples/Form/private_or_posthoc/votes_filled.csv`
- `examples/Form/private_or_posthoc/vote_ties_report.csv`
- `examples/Form/private_or_posthoc/form_option_mapping.json`
- `examples/Form/private_or_posthoc/selected_doctor_by_image.json`

Recommended vote aggregation policy:

- Majority vote per `image_id`.
- Tie-break by random choice with fixed seed.
- Keep a tie report for auditability.

## Label IDs

- `0`: background
- `1`: Epithelial tissue
- `2`: Slough
- `3`: Granulation tissue
- `4`: Necrotic tissue
- `5`: Other
- `255`: ignore
