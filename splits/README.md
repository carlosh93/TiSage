# Dataset Splits

This folder contains the split files used for the TiSage experiments. The repository structure will be further cleaned and documented after the rebuttal phase, but these files are provided here to make the experimental splits explicit and our results fully reproducible.

The files follow the UniMatch-V2 convention:

- `labeled.txt`: image-mask pairs used as labeled training data.
- `unlabeled.txt`: image paths used as unlabeled training data.
- `val.txt`: image-mask pairs used for validation/evaluation.

All paths are relative to the corresponding dataset root.

## Split Generation

The splits were generated with a fixed random seed (`seed=0`) using the scripts
included in this folder. The provided split files should be treated as the
authoritative splits used for the reported experiments.

DFUTissue:

The released DFUTissue files are the paper splits: `fixed`, `1_4`, `1_8`,
and `1_16`. Use the committed files directly for exact reproduction; the split
script is included for transparency.

LUTSeg:

```bash
python prepare_lutseg_splits.py --fixed-split
```

For DFUTissue, fractional splits sample labeled images from the labeled pool and
use the same 600-image unlabeled pool. For LUTSeg, fractional splits sample
labeled images from the training set and use the remaining training images as
unlabeled data.

## Split Sizes

| Dataset | Split | Labeled | Unlabeled | Validation |
| --- | ---: | ---: | ---: | ---: |
| DFUTissue | fixed | 94 | 600 | 16 |
| DFUTissue | 1/4 | 23 | 600 | 16 |
| DFUTissue | 1/8 | 11 | 600 | 16 |
| DFUTissue | 1/16 | 5 | 600 | 16 |
| LUTSeg | fixed | 111 | 111 | 30 |
| LUTSeg | 1/4 | 27 | 84 | 30 |
| LUTSeg | 1/8 | 13 | 98 | 30 |
| LUTSeg | 1/16 | 6 | 105 | 30 |
