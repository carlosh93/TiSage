---
library_name: pytorch
pipeline_tag: image-segmentation
license: apache-2.0
datasets:
  - ksanchez84/LUTSeg
tags:
  - medical
  - semantic-segmentation
  - semi-supervised-learning
  - dinov2
  - medsiglip
---

# TiSage Segmentation Checkpoints

Trained checkpoints associated with the paper settings for **TiSage: Tissue
Segmentation with Multi-Scale Semantic Guidance**, released with the code and reproducibility material at
[carlosh93/TiSage](https://github.com/carlosh93/TiSage). The paper was selected
as a **Spotlight** at the Eleventh ISIC Skin Image Analysis Workshop @ MICCAI
2026.

## Files

| Checkpoint | Dataset | Method | Paper setting |
|---|---|---|---|
| `lutseg_tisage_1_8_seed0_best.pth` | LUTSeg | TiSage | 1/8 labeled, seed 0 |
| `lutseg_unimatch_v2_1_8_seed0_best.pth` | LUTSeg | UniMatch-V2 | 1/8 labeled, seed 0 |
| `dfutissue_tisage_fixed_seed2_best.pth` | DFUTissue | TiSage | fixed/full supervision split, seed 2 |
| `dfutissue_unimatch_v2_fixed_seed1_best.pth` | DFUTissue | UniMatch-V2 | fixed/full supervision split, seed 1 |

Each file contains `model`, `model_ema`, optimizer state, epoch, and validation
selection metadata. The files are best-student snapshots and contain the EMA
state from the same epoch. The paper's Table 1 numbers are selected best-EMA
values across training and remain traceable to the code repository's committed
logs, so an individual released snapshot can differ from the Table 1 value.
The LUTSeg pair is also used by the Figure 4 reproduction script.

## Download

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="ksanchez84/TiSage",
    filename="lutseg_tisage_1_8_seed0_best.pth",
    local_dir="method/checkpoints/downloaded",
)
print(checkpoint)
```

Download every released checkpoint with:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ksanchez84/TiSage",
    local_dir="method/checkpoints/downloaded",
)
```

Verify the files with `segmentation_checkpoints.sha256` from this model
repository or the TiSage code repository.

## Evaluation

Clone the [TiSage repository](https://github.com/carlosh93/TiSage), install its
requirements, download LUTSeg, and run:

```bash
python method/eval/evaluate_checkpoint.py \
  --config method/configs/tisage_lutseg.yaml \
  --checkpoint method/checkpoints/downloaded/lutseg_tisage_1_8_seed0_best.pth
```

The public LUTSeg release has a patient-disjoint validation set and no separate
public test split. The evaluator reports per-class IoU and Dice plus their
means on the 30-image validation set and requires one CUDA GPU. Append
`--check-only` to validate checkpoint compatibility without inference. The
evaluator defaults to the selected student state; pass `--state model_ema` for
the contemporaneous EMA state.

## Architecture and Training

The segmentation network is a DINOv2-Base DPT model. TiSage trains it in an EMA
teacher-student framework using frozen MedSigLIP semantic guidance and the
small dataset-specific prior heads committed in the code repository. MedSigLIP
parameters are not included in these checkpoint files.

Exact configurations, selected seeds, launch commands, result evidence, and
the bounded training check are maintained in the TiSage repository. LUTSeg is
available at [ksanchez84/LUTSeg](https://huggingface.co/datasets/ksanchez84/LUTSeg).

## Intended Use

These checkpoints support research reproduction and non-clinical exploration
of wound-tissue segmentation. They are not medical devices and must not be used
alone for diagnosis or treatment. Results may not transfer to other patient
populations, institutions, cameras, wound etiologies, or acquisition settings.

## License and Attribution

The checkpoint release is distributed under Apache License 2.0 because it
contains DINOv2-derived backbone parameters. TiSage code remains MIT licensed,
and datasets retain their own terms. Training used the gated
[`google/medsiglip-448`](https://huggingface.co/google/medsiglip-448) model as a
frozen prior; its parameters are not redistributed here. See `NOTICE.md` for
the complete attribution and scope.

## Citation

Please cite *LUTSeg: A Longitudinal Multi-Expert Dataset for Ulcer Tissue
Segmentation*, Eleventh ISIC Skin Image Analysis Workshop @ MICCAI 2026. Final
proceedings metadata will be added when the bibliographic record is public.
