# TiSage
Code for Semi-Supervised Tissue Segmentation with Multi-Scale Semantic Guidance

We selected some images and show it here. We don't upload the whole dataset for anonymization reasons as some images contains logos of the participant institutions. if the paper is accepted we will upload the dataset here o in huggingface

Our goal is that reproducibility help researchers create their own dataset for tissue segmentation.

# ICIP CO2 Wounds (Draft README)

This repository contains:

- Code to convert **Label Studio** exports into a train-ready segmentation dataset (LUTS).
- Code to generate **anonymized review-form images** for expert voting (golden set).
- Scripts for inter-rater analysis and visualization.
- (Planned) code for the proposed method and experiments.

The main goal is reproducibility: from raw annotation exports to `train.txt`/`val.txt`, plus a documented expert-review workflow that selects one annotator per image.

## Repository Layout (Key Paths)

- `data/LUTS/pipeline/`: LUTS preprocessing pipeline (steps 1-7) and utilities.
- `data/LUTS/Annotations/raw/`: raw Label Studio export JSONs (input to pipeline).
- `data/LUTS/Annotations/processed/`: processed pipeline artifacts (generated).
- `data/LUTS/`: exported dataset layout (generated): `Images/`, `Masks/`, `Wound_Masks/`, `train.txt`, `val.txt`.
- `data/Dataset_evolution_wounds_VR/`: source image tree expected by the pipeline (not generated here).
- `docs/`: documentation (this file).

## Dataset Labels

Segmentation masks are single-channel PNGs with these class IDs:

- `0`: background
- `1`: Epithelial tissue
- `2`: Slough
- `3`: Granulation tissue
- `4`: Necrotic tissue
- `5`: Other
- `255`: ignore

The visualization palette is defined in `data/LUTS/pipeline/common.py`.

## Quickstart (Generate LUTS Dataset)

From the repository root:

```bash
venv/bin/python data/LUTS/pipeline/run_pipeline.py --default-doctor user_9
```

This runs:

1. Normalize Label Studio exports
2. Rasterize polygons into index masks
3. Group “same image” across annotator folders
4. Select one annotator per image (controlled by a mapping JSON)
5. Patient-level train/val split
6. Export dataset layout to `data/LUTS/`
7. QC report

Notes:

- Step 4 uses `data/LUTS/Annotations/processed/selected_doctor_by_image.json`.
- If that file is missing, `run_pipeline.py` auto-creates it by preferring `--default-doctor` when available, else the first available annotator.
- The split is patient-level. By default, validation patients are a subset of the golden-set patients.

For more details on each step, see `data/LUTS/pipeline/README.md`.

## Train/Val Split Policy

Splits are patient-level to avoid leakage across timepoints for the same patient.

- `data/LUTS/train.txt`: `image_path mask_path` pairs for training
- `data/LUTS/val.txt`: `image_path mask_path` pairs for validation/holdout

By default, the validation set is chosen from a “golden-set” list of patients (see `data/LUTS/pipeline/luts_05_build_splits.py`).

## Expert Voting Workflow (Google Form)

We generate anonymized composite images for each golden-set image:

- Panel 1: Original image
- Panels 2-6: Option A/B/C/D/E (overlay from different annotators, shuffled per image)
- A legend strip is included in the composite.

### 1) Generate Composite Form Images

```bash
venv/bin/python data/LUTS/pipeline/luts_generate_form_images.py
```

Outputs (default):

- `data/LUTS/Annotations/processed/form_review/images/img_XXXX.png`
- `data/LUTS/Annotations/processed/form_review/form_index.csv`
- `data/LUTS/Annotations/processed/form_review/form_option_mapping.json`
- `data/LUTS/Annotations/processed/form_review/votes_template.csv`

Important:

- `form_option_mapping.json` links option letters to annotators. Keep it private until after voting is complete.
- `votes_template.csv` is safe to share with reviewers (it contains no doctor identity).

### 2) Create a Google Form From `form_index.csv` (Apps Script)

High-level steps:

1. Upload the composite images folder to Google Drive:
   - `form_review/images/*.png`
2. Import `form_index.csv` into a Google Sheet (one row per image_id).
3. Attach an Apps Script to the sheet and generate a Google Form that:
   - Adds one image item per `image_id`
   - Adds one required multiple-choice question per `image_id` with options `A,B,C,D,E` (and optional `Skip`)

Example Apps Script (paste into `Extensions -> Apps Script` from the Google Sheet):

```javascript
const CONFIG = {
  FORM_REVIEW_PATH: "2026/MICCAI/form_review", // My Drive path
  INDEX_SHEET_NAME: "form_index",              // change if needed
  FORM_TITLE: "LUTS Segmentation Review",
  ADD_SKIP_OPTION: true,
};

function createLutsReviewFormFromIndex() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const sheet = ss.getSheetByName(CONFIG.INDEX_SHEET_NAME) || ss.getActiveSheet();
  const values = sheet.getDataRange().getValues();
  if (values.length < 2) throw new Error("form_index sheet is empty.");

  const header = values[0].map(String);
  const col = indexColumns_(header, ["image_id", "image_key", "option_labels"]);
  const rows = values.slice(1);

  const imagesFolder = getFolderByPath_(CONFIG.FORM_REVIEW_PATH + "/images");
  const fileMap = buildFileMap_(imagesFolder); // filename -> Drive file

  const form = FormApp.create(CONFIG.FORM_TITLE);
  form.setDescription(
    "For each composite image, select the best option (A/B/C/D/E).\n" +
    "Each image includes an Original panel and option overlays."
  );
  form.setProgressBar(true);
  form.setShuffleQuestions(false);
  form.setCollectEmail(false);
  form.setDestination(FormApp.DestinationType.SPREADSHEET, ss.getId());

  form.addTextItem()
    .setTitle("reviewer_id")
    .setHelpText("Enter your identifier (e.g., Dr_1).")
    .setRequired(true);

  let created = 0;
  const missing = [];

  rows.forEach((r) => {
    const imageId = String(r[col.image_id] || "").trim();
    if (!imageId) return;

    const imageKey = String(r[col.image_key] || "").trim();
    const optionLabels = String(r[col.option_labels] || "")
      .split(",")
      .map(s => s.trim())
      .filter(Boolean);

    const fileName = imageId + ".png";
    const file = fileMap[fileName];
    if (!file) {
      missing.push(fileName);
      return;
    }

    form.addImageItem()
      .setTitle(`Image ${imageId}`)
      .setHelpText(imageKey ? `image_key: ${imageKey}` : "")
      .setImage(file.getBlob());

    const q = form.addMultipleChoiceItem();
    const choices = optionLabels.slice();
    if (CONFIG.ADD_SKIP_OPTION) choices.push("Skip");
    q.setTitle(`Select best option for ${imageId}`)
      .setRequired(true)
      .setChoices(choices.map(c => q.createChoice(c)));

    created++;
  });

  Logger.log("Form edit URL: " + form.getEditUrl());
  Logger.log("Form public URL: " + form.getPublishedUrl());
  Logger.log("Questions created for images: " + created);
  if (missing.length) {
    Logger.log("Missing images (" + missing.length + "): " + missing.join(", "));
  }
}

function indexColumns_(header, required) {
  const map = {};
  header.forEach((h, i) => map[h] = i);
  required.forEach(k => {
    if (!(k in map)) throw new Error("Missing required column: " + k);
  });
  return map;
}

function getFolderByPath_(path) {
  const parts = path.split("/").map(s => s.trim()).filter(Boolean);
  let current = DriveApp.getRootFolder();
  for (const part of parts) {
    const it = current.getFoldersByName(part);
    if (!it.hasNext()) throw new Error("Folder not found at path segment: " + part);
    current = it.next();
  }
  return current;
}

function buildFileMap_(folder) {
  const out = {};
  const it = folder.getFiles();
  while (it.hasNext()) {
    const f = it.next();
    out[f.getName()] = f;
  }
  return out;
}
```

### 3) Convert Votes to a Selection Mapping (After Voting)

After collecting votes, export the Form responses to CSV and convert them to the selection mapping expected by step 4.

This repository expects (eventually) a CSV compatible with:

- `image_id`
- `image_key`
- `selected_option` (e.g., `A`, `B`, `C`, `D`, `E`)

Then run:

```bash
venv/bin/python data/LUTS/pipeline/luts_votes_to_selection_map.py \
  --votes-csv data/LUTS/Annotations/processed/form_review/votes_filled.csv
```

This produces/overwrites:

- `data/LUTS/Annotations/processed/selected_doctor_by_image.json`

Then re-run:

```bash
venv/bin/python data/LUTS/pipeline/run_pipeline.py --default-doctor user_9
```

## Inter-rater Analysis (Figure 1)

Compute ICC by tissue and Dice distribution (golden set by default):

```bash
venv/bin/python data/LUTS/pipeline/luts_inter_rater_figure1.py --save-csvs
```

Outputs:

- `data/LUTS/Annotations/processed/inter_rater_figure1.png`
- Optional CSV tables in `data/LUTS/Annotations/processed/`

Dependency: `pingouin` (`pip install pingouin`).

## Visualizations

Convert index masks to color previews (uses the same palette as `luts_visualize.py`):

```bash
venv/bin/python data/LUTS/pipeline/luts_masks_to_rgb.py
```

Interactive/preview visualization:

```bash
venv/bin/python data/LUTS/pipeline/luts_visualize.py --split train --limit 20
```

## Proposed Method (Placeholder)

TODO: document training/inference entrypoints and hyperparameters for the proposed method in this repository.

Suggested items to include here:

- How to install dependencies (exact versions)
- How to train (command, configs, checkpoints)
- How to evaluate (metrics, scripts)
- How to reproduce reported results

## Reproducibility Notes

- Randomization is controlled by explicit seeds in scripts (e.g., option shuffling for the form images).
- Patient-level splits are deterministic given `--seed`.
- Keep reviewer identity separate from the option-to-doctor mapping until after vote collection is complete.

## Data and Privacy

This work uses medical wound images. Before making this repository public:

- Ensure there is no PHI/PII in committed images or metadata.
- Consider storing large image artifacts via Git LFS or releasing them as a separate archive with an explicit data license.

