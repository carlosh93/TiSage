# LUTS Processing Pipeline

- Raw annotations: `LUTSeg/annotations/raw/`
- Processed artifacts: `LUTSeg/annotations/processed/`
- Source images and final exported dataset root: `LUTSeg/data/`

## Expected Input

- Label Studio exports in `LUTSeg/annotations/raw/*.json`
- Source images under `LUTSeg/data/` (patient/image files referenced by the exports)

## Final Output

- `LUTSeg/data/Images/`
- `LUTSeg/data/Masks/`
- `LUTSeg/data/Wound_Masks/`
- `LUTSeg/data/train.txt`
- `LUTSeg/data/val.txt`
- `LUTSeg/data/class_map.json`
- `LUTSeg/annotations/processed/qc_report.json`

## Label IDs

- `0` background
- `1` Epithelial tissue
- `2` Slough
- `3` Granulation tissue
- `4` Necrotic tissue
- `5` Other
- `255` ignore

## Selection Mapping (Step 4)

Step 4 selects one annotator per image using:

- `LUTSeg/annotations/processed/selected_doctor_by_image.json`

Format:

```json
{
  "selections": {
    "Patient_1/P1_T1.jpeg": "user_9"
  }
}
```

Before voting, bootstrap with a preferred annotator:

```bash
python LUTSeg/pipeline/luts_04_init_selection_map.py \
  --groups-json LUTSeg/annotations/processed/image_groups.json \
  --preferred-doctor user_9 \
  --output-json LUTSeg/annotations/processed/selected_doctor_by_image.json
```

After voting, regenerate mapping from filled votes:

```bash
python LUTSeg/pipeline/luts_votes_to_selection_map.py \
  --form-mapping-json LUTSeg/annotations/processed/form_review/form_option_mapping.json \
  --votes-csv LUTSeg/annotations/processed/form_review/votes_filled.csv \
  --output-json LUTSeg/annotations/processed/selected_doctor_by_image.json \
  --strict
```

## Full Reproducible Pipeline

Run from repository root:

```bash
# 1) Normalize raw exports
python LUTSeg/pipeline/luts_01_normalize_exports.py \
  --raw-dir LUTSeg/annotations/raw \
  --output-json LUTSeg/annotations/processed/normalized_annotations.json \
  --summary-json LUTSeg/annotations/processed/normalize_summary.json

# 2) Rasterize polygons -> per-annotator masks
python LUTSeg/pipeline/luts_02_rasterize_masks.py \
  --normalized-json LUTSeg/annotations/processed/normalized_annotations.json \
  --images-root LUTSeg/data \
  --output-dir LUTSeg/annotations/processed/masks_by_annotator \
  --wound-output-dir LUTSeg/annotations/processed/wound_masks_by_annotator \
  --manifest-json LUTSeg/annotations/processed/rasterized_manifest.json

# 3) Group annotations by image
python LUTSeg/pipeline/luts_03_build_image_groups.py \
  --manifest-json LUTSeg/annotations/processed/rasterized_manifest.json \
  --output-json LUTSeg/annotations/processed/image_groups.json

# 4) Select one mask per image
python LUTSeg/pipeline/luts_04_select_masks.py \
  --groups-json LUTSeg/annotations/processed/image_groups.json \
  --selection-json LUTSeg/annotations/processed/selected_doctor_by_image.json \
  --default-doctor user_9 \
  --strict-selection \
  --output-dir LUTSeg/data/Masks \
  --wound-output-dir LUTSeg/data/Wound_Masks \
  --manifest-json LUTSeg/annotations/processed/consensus_manifest.json

# 5) Build train/val split
python LUTSeg/pipeline/luts_05_build_splits.py \
  --consensus-manifest LUTSeg/annotations/processed/consensus_manifest.json \
  --output-json LUTSeg/annotations/processed/splits.json \
  --seed 42

# 6) Export dataset layout
python LUTSeg/pipeline/luts_06_export_dataset_layout.py \
  --splits-json LUTSeg/annotations/processed/splits.json \
  --dataset-root LUTSeg/data \
  --overwrite

# 7) QC report
python LUTSeg/pipeline/luts_07_qc_report.py \
  --consensus-manifest LUTSeg/annotations/processed/consensus_manifest.json \
  --output-json LUTSeg/annotations/processed/qc_report.json
```

Equivalent one-command wrapper:

```bash
python LUTSeg/pipeline/run_pipeline.py --default-doctor user_9
```


## Form Workflow (Golden Set Review)

Generate form composites:

```bash
python LUTSeg/pipeline/luts_generate_form_images.py \
  --groups-json LUTSeg/annotations/processed/image_groups.json \
  --output-dir LUTSeg/annotations/processed/form_review \
  --golden-only \
  --seed 42
```

Outputs:

- `LUTSeg/annotations/processed/form_review/images/img_XXXX.png` (2x3 grid + legend)
- `LUTSeg/annotations/processed/form_review/form_option_mapping.json`
- `LUTSeg/annotations/processed/form_review/form_index.csv`
- `LUTSeg/annotations/processed/form_review/votes_template.csv`

Convert Google Form responses to `votes_filled.csv`:

```bash
python LUTSeg/pipeline/luts_form_responses_to_votes.py \
  --responses-csv "LUTSeg/annotations/raw/LUTSeg Dataset Golden Set Review.csv" \
  --votes-template-csv LUTSeg/annotations/processed/form_review/votes_template.csv \
  --form-mapping-json LUTSeg/annotations/processed/form_review/form_option_mapping.json \
  --output-csv LUTSeg/annotations/processed/form_review/votes_filled.csv \
  --report-ties-csv LUTSeg/annotations/processed/form_review/vote_ties_report.csv \
  --tie-break random \
  --seed 42
```

Then regenerate selection mapping and rerun steps 4-7.

Operational notes:

- After new votes are integrated, you only need to rerun steps 4-7 (not steps 1-3).

## Inter-Rater Analysis

```bash
python LUTSeg/pipeline/luts_inter_rater_figure1.py \
  --groups-json LUTSeg/annotations/processed/image_groups.json \
  --output-figure LUTSeg/annotations/processed/inter_rater_figure1.png \
  --output-dir LUTSeg/annotations/processed \
  --save-csvs
```

Requires `pingouin` package.

## Visualizations and RGB Masks

```bash
python LUTSeg/pipeline/luts_masks_to_rgb.py \
  --masks-dir LUTSeg/data/Masks \
  --output-dir LUTSeg/data/Masks_RGB

python LUTSeg/pipeline/luts_visualize.py \
  --dataset-root LUTSeg/data \
  --split train \
  --no-legend \
  --include-golden \
  --save-dir tmp/tissue_train

python LUTSeg/pipeline/luts_visualize.py \
  --dataset-root LUTSeg/data \
  --split val \
  --no-legend \
  --include-golden \
  --save-dir tmp/tissue_val

python LUTSeg/pipeline/luts_visualize_wound_detection.py \
  --dataset-root LUTSeg/data \
  --split train \
  --include-golden \
  --save-dir tmp/wound_train

python LUTSeg/pipeline/luts_visualize_wound_detection.py \
  --dataset-root LUTSeg/data \
  --split val \
  --include-golden \
  --save-dir tmp/wound_val
```

## Seeds Used

Use seed `42` for reproducibility unless you intentionally change it:

- `luts_generate_form_images.py --seed 42`
- `luts_form_responses_to_votes.py --seed 42` (tie-break random)
- `luts_05_build_splits.py --seed 42`

If you need exact split reproducibility over time, keep `--val-patients` explicit in step 5.
