# Reported Experiment Evidence

This directory contains the sanitized training and evaluation evidence for all
experiments reported in the paper:

- `table1/`: supervised and semi-supervised comparisons plus LUTSeg full-supervision references.
- `table3_prior/`: single-scale and multiscale MedSigLIP prior-only evaluation artifacts.
- `component_ablation/`: DFUTissue 1/8 full method and three component removals for seeds 0–2.
- `sensitivity/`: LUTSeg 1/8 gate and alpha sweeps.

Infrastructure-only output such as local absolute paths, cluster hostnames,
NCCL diagnostics, and W&B run identifiers has been removed. Experiment
arguments, epoch metrics, per-class IoU values, and reported results are
preserved. Run `python method/eval/verify_reported_results.py` from the
repository root to validate every reported value.
