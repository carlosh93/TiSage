# TiSage

Reproducibility repository for Semi-Supervised Tissue Segmentation with Multi-Scale Semantic Guidance (Paper ID: 3280).

## Repository Structure

- `method/`: TiSage training/evaluation code, configs, checkpoints, and linked logs for the main paper table.
- `dataset/`, `model/`, `util/`, `supervised.py`: runtime modules required by `method/src/tisage/train.py`.
- `LUTSeg/`: LUTSeg dataset-construction reproducibility pipeline and examples.

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then see:

- Method reproduction: `method/README.md`
- LUTSeg pipeline/data workflow: `LUTSeg/README.md`

## Data Availability

- The full training dataset is not included in this anonymous review repository because some images contain institutional landmarks that could compromise anonymity. All images were de-identified prior to the study in accordance with institutional ethical guidelines. The complete dataset will be released upon acceptance.
- A small non-identifying example subset is included under `LUTSeg/examples/` to document file formats and the annotation-voting workflow.

## Reproducibility Notes

- Main results table values and source logs are provided in `method/README.md` and `method/logs/main_table/`.
