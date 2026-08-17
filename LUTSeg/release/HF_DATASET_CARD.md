---
pretty_name: LUTSeg
license: cc-by-4.0
task_categories:
  - image-segmentation
tags:
  - medical
  - wound-care
  - semantic-segmentation
  - longitudinal
  - multi-expert
  - leprosy
  - image
size_categories:
  - n<1K
---

# LUTSeg

**LUTSeg: A Longitudinal Multi-Expert Dataset for Ulcer Tissue Segmentation**
contains pixel-level tissue annotations for longitudinal, leprosy-related chronic
ulcer images.

The accompanying TiSage paper was selected as a **Spotlight** at the Eleventh
ISIC Skin Image Analysis Workshop @ MICCAI 2026.

## Dataset Summary

- 141 images from 39 pseudonymized patients
- 111 training images and 30 validation images, split at the patient level
- Longitudinal acquisition over 21 months
- Binary wound masks and six-class tissue masks, including background
- A 46-image gold-standard subset from 9 patients annotated by five clinicians
- Per-clinician masks and inter-rater agreement artifacts for the gold subset

LUTSeg reorganizes images collected in the SIMATEC project by patient and time
and adds new expert tissue labels. The source images originate from the
[CO2Wounds dataset](https://data.mendeley.com/datasets/nkw5gx57hw/1) described
in the [original study](https://doi.org/10.1016/j.compbiomed.2023.107753).

## Labels

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | Epithelial tissue |
| 2 | Slough |
| 3 | Granulation tissue |
| 4 | Necrotic tissue |
| 5 | Other |

`Masks/` stores single-channel tissue IDs. `Wound_Masks/` stores binary masks
with values 0 and 255. `Masks_RGB/` provides visualizations and must not be used
as training targets.

## Repository Layout

```text
Images/                         source RGB images
Masks/                          tissue-label masks
Masks_RGB/                      colorized tissue-mask visualizations
Wound_Masks/                    binary wound masks
metadata.jsonl                  paired files and sample metadata
train.txt, val.txt              full-supervision patient-level split
splits/                         full, 1/4, 1/8, and 1/16 paper splits
gold_standard/                  multi-expert masks and agreement artifacts
checksums.sha256                release integrity manifest
```

The identifiers in paths and metadata are dataset-internal pseudonyms. They are
not hospital identifiers or patient names. Clinician and reviewer identifiers
are also permanent pseudonyms.

## Download and Use with TiSage

Download directly into the location expected by TiSage:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ksanchez84/LUTSeg",
    repo_type="dataset",
    local_dir="data/LUTSeg",
)
```

The resulting `data/LUTSeg/Images`, `data/LUTSeg/Masks`, `train.txt`, and
`val.txt` paths work directly with the code at
[carlosh93/TiSage](https://github.com/carlosh93/TiSage).

For Hugging Face Datasets, `metadata.jsonl` uses multiple `*_file_name` fields
to pair each image with its tissue, visualization, and wound masks. The
`split` column distinguishes training and validation samples.

## Annotation Protocol

Five clinicians with wound-care and skin-tissue expertise used a standardized
interface. Wound boundaries were delineated first, followed by pixel-level
annotation of epithelial, slough, granulation, necrotic, and other tissue. For
the 46-image gold subset, all five clinicians annotated every image. A single
reference mask was selected by anonymized clinician voting; ties used a
fixed-seed random selection. The released inter-rater files support the
paper-reported ICC and pairwise Dice analyses.

## Ethics and Privacy

Acquisition followed the Declaration of Helsinki. All data were anonymized,
written informed consent was obtained from all participants, and the study was
approved by the participating hospitals' ethics committees (Approval Nos.
05-21 and 30-11-25).

The release process removes EXIF, GPS, XMP, comments, and editing metadata from
all images. Visual content should still be treated as sensitive medical data
and handled according to applicable institutional and legal requirements.

## Intended Uses and Limitations

LUTSeg is intended for research in wound tissue segmentation, longitudinal
wound analysis, annotation variability, and label-efficient learning. It is not
a medical device and must not be used alone for diagnosis or treatment.

The dataset is small, represents a specific clinical and disease context, uses
smartphone imagery, and contains substantial class imbalance and inter-rater
variability. Performance may not transfer to other populations, institutions,
cameras, wound etiologies, or care settings without additional validation.

## License

LUTSeg is released under the Creative Commons Attribution 4.0 International
License (CC BY 4.0). Users must provide appropriate attribution and preserve the
dataset citation. This is the same license as the original CO2Wounds source
image release; LUTSeg's new annotations, splits, and metadata are distributed
under the same terms.

## Citation

Please cite *LUTSeg: A Longitudinal Multi-Expert Dataset for Ulcer Tissue
Segmentation*, Eleventh ISIC Skin Image Analysis Workshop @ MICCAI 2026. The
final proceedings BibTeX will be added when the bibliographic record is public.
