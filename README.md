# LUTSeg and TiSage

Official code and reproducibility repository for **LUTSeg: A Longitudinal
Multi-Expert Dataset for Ulcer Tissue Segmentation** and the TiSage
semi-supervised tissue-segmentation method.

> **Spotlight paper — Eleventh ISIC Skin Image Analysis Workshop @ MICCAI 2026**

TiSage builds on an EMA teacher-student segmentation framework and introduces
multi-scale MedSigLIP semantic guidance, pixel-adaptive fusion, and
confidence-weighted soft supervision for low-label wound-tissue segmentation.

## Repository Contents

- `method/`: TiSage code, paper configurations, launchers, result verification, figures, and evidence logs.
- `baselines/`: FixMatch, UniMatch-V2, and DeepLabV3+ implementations used in Table 1.
- `dataset/`, `model/`, `util/`, `supervised.py`: shared training runtime required by TiSage and the baselines.
- `splits/`: exact DFUTissue and LUTSeg split files used in the paper.
- `LUTSeg/`: dataset-construction pipeline, pseudonymized expert annotations, examples, and inter-rater analysis.

Only experiments reported in the paper are included: Tables 1–3, the
DFUTissue component ablation, the LUTSeg sensitivity analysis, Figure 4, and
the LUTSeg inter-rater analysis.

## Installation

Python 3.10 is recommended.

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The experiments were validated with Python 3.10.19, PyTorch 2.6.0,
torchvision 0.21.0, CUDA 12.4, and Transformers 5.1.0. Install the PyTorch
build appropriate for your CUDA driver before installing the remaining
requirements when using a different CUDA version. Weights & Biases is optional
and can be enabled after installing `wandb` separately.

Confirm the entrypoint and committed results:

```bash
python method/src/tisage/train.py --help
python method/eval/verify_reported_results.py
```

For a minimal end-user check, one command downloads LUTSeg, validates its
layout and patient-level split, loads a sample through TiSage, checks the
training entrypoint and result evidence, and performs a launcher dry run:

```bash
python smoke_test.py --download
```

This test does not require a GPU and does not launch training. After the first
download, rerun it without `--download`.

## Data

Expected training directories are:

```text
data/
├── DFUTissue/
└── LUTSeg/
    ├── Images/
    ├── Masks/
    ├── train.txt
    └── val.txt
```

The complete LUTSeg dataset is public at
[ksanchez84/LUTSeg](https://huggingface.co/datasets/ksanchez84/LUTSeg). Download
it directly into the path expected by the training code:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ksanchez84/LUTSeg', repo_type='dataset', local_dir='data/LUTSeg')"
```

A small pseudonymized example subset is also available in `LUTSeg/examples/`.
LUTSeg reorganizes the CC BY 4.0 SIMATEC/CO2Wounds source images by patient and
time and adds new expert tissue annotations under the same license.

See `LUTSeg/README.md` for dataset construction and `splits/README.md` for the
exact experimental partitions.

## Pretrained Weights

- Download the official DINOv2-Base weights to the path expected by TiSage:

```bash
mkdir -p pretrained
python -c "from urllib.request import urlretrieve; urlretrieve('https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth', 'pretrained/dinov2_base.pth')"
```

- TiSage loads the gated [`google/medsiglip-448`](https://huggingface.co/google/medsiglip-448)
  model on first use. Accept its access terms and run `hf auth login` before training.
- Place ImageNet ResNet-50 weights at `pretrained/resnet50-11ad3fa6.pth` for DeepLabV3+.
- The small MedSigLIP prior heads are included in `method/checkpoints/pretrained/`.
- Four trained segmentation checkpoints associated with the paper settings are released at
  [ksanchez84/TiSage](https://huggingface.co/ksanchez84/TiSage). They are not
  duplicated in Git because each is approximately 866 MB. Download and
  integrity-check instructions are in `method/checkpoints/`.

## Quick Training Check

The following is the smallest real TiSage training check. It loads LUTSeg and
both pretrained models, performs two optimizer steps, evaluates all 30
validation images, and writes `latest.pth`, `best.pth`, TensorBoard events, and
the console log to a temporary directory. It does not reproduce a paper score.

```bash
set -o pipefail
SMOKE_DIR="$(mktemp -d /tmp/tisage-train-smoke.XXXXXX)"
python method/scripts/run_table1.py \
  --method tisage \
  --dataset lutseg \
  --split 1_16 \
  --nproc-per-node 1 \
  --epochs 1 \
  --max-steps-per-epoch 2 \
  --save-path "$SMOKE_DIR" \
  2>&1 | tee "$SMOKE_DIR/train.log"
echo "Artifacts: $SMOKE_DIR"
```

This requires one CUDA GPU and approximately 1.7 GB of temporary disk space for
the two checkpoints. Add `--no-save` for a disk-light check. Delete the printed
temporary directory after inspecting it.

LUTSeg provides training and validation partitions, not a separate test split.
The command above therefore performs the repository's quantitative validation
step. To evaluate a released checkpoint independently, see
`method/eval/evaluate_checkpoint.py` and the commands in `method/README.md`.

## Reproducing Results

The complete commands, linked result tables, and evaluation protocol are in
`method/README.md`. For example:

```bash
python method/scripts/run_table1.py \
  --method tisage --dataset lutseg --split 1_8 --dry-run

python method/scripts/run_component_ablation.py \
  --variant full --seed 0 --dry-run

python method/scripts/run_sensitivity.py \
  --parameter alpha --value 0.25 --dry-run
```

Remove `--dry-run` to launch an experiment. Training requires an NVIDIA GPU.
The reported DINOv2–DPT, FixMatch, UniMatch-V2, TiSage, ablation, and
sensitivity runs used two GPUs; DeepLabV3+ used one. These are the launcher
defaults. Use `--nproc-per-node 1` to run any configuration on one GPU.

## Citation

The final author list and proceedings identifier will be added after the
camera-ready bibliographic record becomes public. Until then, please cite the
paper by title and the Eleventh ISIC Skin Image Analysis Workshop @ MICCAI 2026.

## License

Code is released under the MIT License. LUTSeg data is released under CC BY
4.0; see `LUTSeg/DATA_LICENSE.md`. Upstream attributions are listed
in `THIRD_PARTY_NOTICES.md`.
