# Examples

This folder contains a curated subset of images generated with the scripts in `LUTSeg/pipeline/`

## Contents

- `Images/`: curated sample input images.
- `Masks/`: sample tissue masks aligned to `Images/`.
- `Wound_Masks/`: sample wound masks aligned to `Images/`.
- `Masks_RGB/`: RGB-rendered versions of sample tissue masks.
- `visualizations/`: Visualization outputs corresponding to masks on top of images.
- `Form/`: form assets and post-vote outputs.

## Form Subfolders

- `Form/public/`: safe-to-share artifacts (`images/`, `form_index.csv`, `votes_template.csv`).
- `Form/private_or_posthoc/`: mapping and post-vote files (`form_option_mapping.json`, votes, tie report, final selection mapping).

## Example images

Core example assets (`Images/`, `Masks/`, `Wound_Masks/`, `Masks_RGB/`, `visualizations/`) are generated from:

- Patient_10: `P10T_1` to `P10T_3`
- Patient_34: `P34T_1` to `P34T_4`
- Patient_20: all except `P20T_5`

Form assets (`Form/public`, `Form/private_or_posthoc`) are restricted to golden/validation images from:

- Patient_20
- Patient_5

## Anonymization

- Patient-style identifiers (e.g., `Patient_20`, `P20T_1`) are dataset-internal codes, not personal identifiers.
- Annotator identifiers in form artifacts use pseudonymous IDs (e.g., `user_7`).
