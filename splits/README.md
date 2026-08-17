# Dataset Splits

These are the exact split files used for the paper experiments. Paths are
relative to the corresponding dataset root.

- `labeled.txt`: image-mask pairs used as labeled training data.
- `unlabeled.txt`: image paths used as unlabeled training data.
- `val.txt`: image-mask pairs used for validation and evaluation.

## Split Sizes

| Dataset | Public name | Directory | Labeled | Unlabeled | Validation |
|---|---|---|---:|---:|---:|
| DFUTissue | Fixed | `fixed` | 94 | 600 | 16 |
| DFUTissue | 1/4 | `1_4` | 23 | 600 | 16 |
| DFUTissue | 1/8 | `1_8` | 11 | 600 | 16 |
| DFUTissue | 1/16 | `1_16` | 5 | 600 | 16 |
| LUTSeg | Full | `fixed` | 111 | 111 | 30 |
| LUTSeg | 1/4 | `1_4` | 27 | 84 | 30 |
| LUTSeg | 1/8 | `1_8` | 13 | 98 | 30 |
| LUTSeg | 1/16 | `1_16` | 6 | 105 | 30 |

LUTSeg `fixed` is the internal directory name for the 111-image full-supervision
reference. Portable launchers accept the public name `full` and map it to this
directory.

## Generation

The committed files are authoritative for exact reproduction. The generation
scripts are included for transparency and use fixed random seeds:

```bash
python splits/prepare_dfutissue.py --ratios 4,8,16 --fixed-split
python splits/prepare_lutseg_splits.py --ratios 4,8,16 --fixed-split
```
