# LUTSeg Dataset Reproducibility

This directory documents construction of LUTSeg from pseudonymized Label Studio
exports, multi-expert annotations, and the clinician-voting workflow used for
gold-set selection.

## Contents

- `pipeline/`: normalization, rasterization, consensus, split, quality-control, visualization, and inter-rater scripts.
- `annotations/raw/`: pseudonymized Label Studio exports used by the construction pipeline.
- `examples/`: a small dataset-layout example and pseudonymized voting artifacts.
- `inter_rater_figure1.png`: the paper-reported inter-rater analysis figure.
- `DATA_LICENSE.md`: CC BY 4.0 terms and source-dataset provenance.

The full dataset contains 141 longitudinal images from 39 patients, with 111
training images and 30 validation images. The gold subset contains 46 images
from 9 patients annotated by five clinicians. All patient and clinician
identifiers in this repository are pseudonymous dataset codes.

## Data Availability

The complete LUTSeg release is public at
[ksanchez84/LUTSeg](https://huggingface.co/datasets/ksanchez84/LUTSeg). Download
it into the TiSage dataset location with:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ksanchez84/LUTSeg', repo_type='dataset', local_dir='data/LUTSeg')"
```

The examples committed here document formats and workflows but are not a
replacement for the complete training dataset.

The deterministic release builder in `release/` creates the upload-ready
dataset without modifying the source files. It strips embedded image metadata,
selects the documented gold-standard annotations, generates checksums, and
validates privacy and data integrity before publication.

## Reproduced Workflow

1. Normalize Label Studio polygon exports.
2. Rasterize tissue and wound-outline masks per annotator.
3. Group annotations by image.
4. Generate clinician-comparison forms and aggregate pseudonymized votes.
5. Select the final mask for each image.
6. Build patient-level train/validation splits and export the dataset layout.
7. Generate quality-control and inter-rater reports.

Run the pipeline from the repository root:

```bash
python LUTSeg/pipeline/run_pipeline.py --default-doctor user_9
```

See `pipeline/README.md` for individual commands and intermediate artifacts.

## Voting Artifacts

- Public comparison forms: `examples/Form/public/`
- Pseudonymized responses and selections: `examples/Form/anonymized_votes/`

The aggregation policy uses majority vote per image and a fixed-seed random
tie-break while retaining a tie report for auditability.

## Label IDs

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | Epithelial tissue |
| 2 | Slough |
| 3 | Granulation tissue |
| 4 | Necrotic tissue |
| 5 | Other |
| 255 | Ignore |
