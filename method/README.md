# TiSage Method

This folder contains the method code and experiment scripts used to train, analyze, and report TiSage results.

## Directory Layout

- `src/tisage/`: main training entrypoint and MedSigLIP prior calibrator.
- `configs/`: DFUTissue and LUTSeg training configs.
- `scripts/`: prior training and prior-only evaluations.
- `eval/`: scripts to extract table and per-class metrics from experiment logs.
- `checkpoints/`: expected location of prior classifier checkpoints.
- `logs/`: logs supporting reproduced table values.

## Required Inputs

This repository is self-contained for the TiSage method code. The project root (`tmp/TiSage`) must contain:

- `data/DFUTissue` and `data/LUTSeg`
- `splits/dfutissue/...` and `splits/lutseg/...`
- DINOv2 backbone weights under `pretrained/` (for segmentation training)
- Training support modules at project root: `dataset/`, `model/`, `util/`, and `supervised.py`

For MedSigLIP prior integration, expected default classifier checkpoints are:

- `method/checkpoints/pretrained/medsiglip_head_dfutissue.pt`
- `method/checkpoints/pretrained/medsiglip_head_lutseg.pt`

## Main Table (Linked Logs)

LUTSeg full-supervision reference: [31.37 / 39.19*](logs/main_table/labeled_only/lutseg/full_reference/out.log)
All linked logs in `logs/main_table/` are sanitized for anonymized review.

| Method | DFUTissue Fixed mIoU | DFUTissue Fixed F1 | DFUTissue 1/4 mIoU | DFUTissue 1/4 Dice | DFUTissue 1/8 mIoU | DFUTissue 1/8 Dice | DFUTissue 1/16 mIoU | DFUTissue 1/16 Dice | LUTSeg 1/4 mIoU | LUTSeg 1/4 Dice | LUTSeg 1/8 mIoU | LUTSeg 1/8 Dice | LUTSeg 1/16 mIoU | LUTSeg 1/16 Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Labeled Only | [68.71](logs/main_table/labeled_only/dfutissue/fixed/out.log) | [80.21](logs/main_table/labeled_only/dfutissue/fixed/out.log) | [66.23](logs/main_table/labeled_only/dfutissue/1_4/out.log) | [78.03](logs/main_table/labeled_only/dfutissue/1_4/out.log) | [64.68](logs/main_table/labeled_only/dfutissue/1_8/out.log) | [76.57](logs/main_table/labeled_only/dfutissue/1_8/out.log) | [52.83](logs/main_table/labeled_only/dfutissue/1_16/out.log) | [65.02](logs/main_table/labeled_only/dfutissue/1_16/out.log) | [29.38](logs/main_table/labeled_only/lutseg/1_4/out.log) | [35.19](logs/main_table/labeled_only/lutseg/1_4/out.log) | [20.47](logs/main_table/labeled_only/lutseg/1_8/out.log) | [23.55](logs/main_table/labeled_only/lutseg/1_8/out.log) | [24.47](logs/main_table/labeled_only/lutseg/1_16/out.log) | [30.15](logs/main_table/labeled_only/lutseg/1_16/out.log) |
| FixMatch | [68.91](logs/main_table/fixmatch/dfutissue/fixed/out.log) | [80.19](logs/main_table/fixmatch/dfutissue/fixed/out.log) | [67.17](logs/main_table/fixmatch/dfutissue/1_4/out.log) | [78.80](logs/main_table/fixmatch/dfutissue/1_4/out.log) | [66.90](logs/main_table/fixmatch/dfutissue/1_8/out.log) | [78.40](logs/main_table/fixmatch/dfutissue/1_8/out.log) | [60.14](logs/main_table/fixmatch/dfutissue/1_16/out.log) | [71.30](logs/main_table/fixmatch/dfutissue/1_16/out.log) | [27.70](logs/main_table/fixmatch/lutseg/1_4/out.log) | [33.00](logs/main_table/fixmatch/lutseg/1_4/out.log) | [27.26](logs/main_table/fixmatch/lutseg/1_8/out.log) | [33.91](logs/main_table/fixmatch/lutseg/1_8/out.log) | [27.42](logs/main_table/fixmatch/lutseg/1_16/out.log) | [34.33](logs/main_table/fixmatch/lutseg/1_16/out.log) |
| UniMatch-V2 | [69.94](logs/main_table/unimatch_v2/dfutissue/fixed/out.log) | [80.96](logs/main_table/unimatch_v2/dfutissue/fixed/out.log) | [68.17](logs/main_table/unimatch_v2/dfutissue/1_4/out.log) | [79.67](logs/main_table/unimatch_v2/dfutissue/1_4/out.log) | [67.28](logs/main_table/unimatch_v2/dfutissue/1_8/out.log) | [78.85](logs/main_table/unimatch_v2/dfutissue/1_8/out.log) | [61.80](logs/main_table/unimatch_v2/dfutissue/1_16/out.log) | [73.24](logs/main_table/unimatch_v2/dfutissue/1_16/out.log) | [26.13](logs/main_table/unimatch_v2/lutseg/1_4/out.log) | [30.55](logs/main_table/unimatch_v2/lutseg/1_4/out.log) | [27.60](logs/main_table/unimatch_v2/lutseg/1_8/out.log) | [34.24](logs/main_table/unimatch_v2/lutseg/1_8/out.log) | [27.35](logs/main_table/unimatch_v2/lutseg/1_16/out.log) | [32.24](logs/main_table/unimatch_v2/lutseg/1_16/out.log) |
| **TiSage (Ours)** | [**72.36**](logs/main_table/tisage/dfutissue/fixed/out.log) | [**83.05**](logs/main_table/tisage/dfutissue/fixed/out.log) | [**69.77**](logs/main_table/tisage/dfutissue/1_4/out.log) | [**81.00**](logs/main_table/tisage/dfutissue/1_4/out.log) | [**67.93**](logs/main_table/tisage/dfutissue/1_8/out.log) | [**79.28**](logs/main_table/tisage/dfutissue/1_8/out.log) | [61.33](logs/main_table/tisage/dfutissue/1_16/out.log) | [73.17](logs/main_table/tisage/dfutissue/1_16/out.log) | [**28.73**](logs/main_table/tisage/lutseg/1_4/out.log) | [**34.50**](logs/main_table/tisage/lutseg/1_4/out.log) | [**31.70**](logs/main_table/tisage/lutseg/1_8/out.log) | [**39.25**](logs/main_table/tisage/lutseg/1_8/out.log) | [**28.55**](logs/main_table/tisage/lutseg/1_16/out.log) | [**34.04**](logs/main_table/tisage/lutseg/1_16/out.log) |

## Environment

At minimum, install:

```bash
pip install torch torchvision transformers scikit-image pyyaml tensorboard
```

Optional:

```bash
pip install wandb
```

## Reproduction Workflow

Run from the repository root (`tmp/TiSage`):

1. (Optional) Train prior heads
```bash
python3 method/scripts/train_prior_dfutissue.py
python3 method/scripts/train_prior_lutseg.py
```

2. Run TiSage training (DFUTissue)
```bash
torchrun --standalone --nproc_per_node=1 method/src/tisage/train.py \
  --config method/configs/tisage_dfutissue.yaml \
  --labeled-id-path splits/dfutissue/1_8/labeled.txt \
  --unlabeled-id-path splits/dfutissue/1_8/unlabeled.txt \
  --save-path exp/dfutissue/tisage_medsiglip/seed0 \
  --seed 0 \
  --medsiglip \
  --medsiglip-classifier-path method/checkpoints/pretrained/medsiglip_head_dfutissue.pt
```

3. Run TiSage training (LUTSeg)
```bash
torchrun --standalone --nproc_per_node=1 method/src/tisage/train.py \
  --config method/configs/tisage_lutseg.yaml \
  --labeled-id-path splits/lutseg/1_8/labeled.txt \
  --unlabeled-id-path splits/lutseg/1_8/unlabeled.txt \
  --save-path exp/lutseg/tisage_medsiglip/seed0 \
  --seed 0 \
  --medsiglip \
  --medsiglip-classifier-path method/checkpoints/pretrained/medsiglip_head_lutseg.pt
```

4. Extract report tables from `exp/`
```bash
python3 method/eval/extract_main_table.py --help
python3 method/eval/extract_perclass_table.py --help
```

## Notes

- Increase `--nproc_per_node` for multi-GPU training on one node.
- Prior-only utilities in `method/scripts/` can be run independently.
