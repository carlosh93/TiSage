#!/usr/bin/env bash
# Ablation over coarse/fine superpixel params and fusion beta for LUTSeg
# using MedSigLIP + classifier priors only.
#
# Usage (from repo root):
#   bash method/scripts/ablate_prior_multiscale_lutseg.sh
#   CLASSIFIER_PATH=/abs/path/medsiglip_head_lutseg.pt bash method/scripts/ablate_prior_multiscale_lutseg.sh
#
# Optional env:
#   PYTHON_BIN, DATA_ROOT, VAL_TXT, OUTPUT_DIR, RUN_TAG
#   CROP_MODE, OUTSIDE_FILL, SMALL_REGION_RATIO_THRESH, SMALL_REGION_ZOOM
#   COARSE_N_ARR, COARSE_MIN_ARR, FINE_N_ARR, FINE_MIN_ARR, BETA_ARR (comma-separated)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$METHOD_DIR/.." && pwd)"
cd "$METHOD_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
CLASSIFIER_PATH="${CLASSIFIER_PATH:-$METHOD_DIR/checkpoints/pretrained/medsiglip_head_lutseg.pt}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/LUTSeg}"
VAL_TXT="${VAL_TXT:-$PROJECT_ROOT/splits/lutseg/val.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results_medsiglip_multiscale_ablation_lutseg}"
CROP_MODE="${CROP_MODE:-bbox}"
OUTSIDE_FILL="${OUTSIDE_FILL:-mean}"
SMALL_REGION_RATIO_THRESH="${SMALL_REGION_RATIO_THRESH:-0.0}"
SMALL_REGION_ZOOM="${SMALL_REGION_ZOOM:-1.0}"
RUN_TAG="${RUN_TAG:-mode_${CROP_MODE}_c64_f192_40_beta07_08_09_$(date +%Y%m%d_%H%M%S)}"
TSV_PATH="$OUTPUT_DIR/ablation_results_${RUN_TAG}.tsv"
SUMMARY_PATH="$OUTPUT_DIR/top_configs_${RUN_TAG}.txt"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: PYTHON_BIN not executable: $PYTHON_BIN"
  exit 1
fi
if [[ ! -f "$CLASSIFIER_PATH" ]]; then
  echo "ERROR: Classifier not found: $CLASSIFIER_PATH"
  echo "Train first with method/scripts/train_prior_lutseg.py or set CLASSIFIER_PATH."
  exit 1
fi
if [[ ! -f "$VAL_TXT" ]]; then
  echo "ERROR: Val split not found: $VAL_TXT"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
rm -f "$TSV_PATH"

# Focused test around the current hypothesis.
IFS=',' read -r -a coarse_n_arr <<< "${COARSE_N_ARR:-64}"
IFS=',' read -r -a coarse_min_arr <<< "${COARSE_MIN_ARR:-100}"
IFS=',' read -r -a fine_n_arr <<< "${FINE_N_ARR:-192}"
IFS=',' read -r -a fine_min_arr <<< "${FINE_MIN_ARR:-40}"
IFS=',' read -r -a beta_arr <<< "${BETA_ARR:-0.7,0.8,0.9}"

total=0
for coarse_n in "${coarse_n_arr[@]}"; do
  for coarse_min in "${coarse_min_arr[@]}"; do
    for fine_n in "${fine_n_arr[@]}"; do
      for fine_min in "${fine_min_arr[@]}"; do
        for beta in "${beta_arr[@]}"; do
          total=$((total + 1))
        done
      done
    done
  done
done

echo "Running $total LUTSeg ablation runs..."
echo "  classifier: $CLASSIFIER_PATH"
echo "  run tag:    $RUN_TAG"
echo "  output TSV: $TSV_PATH"
echo "  crop mode:  $CROP_MODE"
echo "  fill mode:  $OUTSIDE_FILL"
echo "  small zoom: thresh=$SMALL_REGION_RATIO_THRESH zoom=$SMALL_REGION_ZOOM"

n=0
for coarse_n in "${coarse_n_arr[@]}"; do
  for coarse_min in "${coarse_min_arr[@]}"; do
    for fine_n in "${fine_n_arr[@]}"; do
      for fine_min in "${fine_min_arr[@]}"; do
        for beta in "${beta_arr[@]}"; do
          n=$((n + 1))
          echo "[$n/$total] coarse=${coarse_n}/${coarse_min} fine=${fine_n}/${fine_min} beta=$beta"
          "$PYTHON_BIN" scripts/eval_prior_multiscale_lutseg.py \
            --classifier-path "$CLASSIFIER_PATH" \
            --data-root "$DATA_ROOT" \
            --val-txt "$VAL_TXT" \
            --coarse-n-segments "$coarse_n" \
            --coarse-min-size "$coarse_min" \
            --fine-n-segments "$fine_n" \
            --fine-min-size "$fine_min" \
            --beta "$beta" \
            --crop-mode "$CROP_MODE" \
            --outside-fill "$OUTSIDE_FILL" \
            --small-region-ratio-thresh "$SMALL_REGION_RATIO_THRESH" \
            --small-region-zoom "$SMALL_REGION_ZOOM" \
            --results-tsv "$TSV_PATH" \
            --quiet
        done
      done
    done
  done
done

echo ""
echo "Ranking configurations by fused mIoU..."
"$PYTHON_BIN" - "$TSV_PATH" "$SUMMARY_PATH" << 'PY'
import csv
import sys

tsv_path = sys.argv[1]
summary_path = sys.argv[2]

with open(tsv_path, newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

if not rows:
    print("No rows found in TSV.")
    sys.exit(0)

def fnum(v, default=-1e9):
    try:
        return float(v)
    except Exception:
        return default

rows_sorted = sorted(rows, key=lambda r: fnum(r.get("fused_miou")), reverse=True)
top_k = min(10, len(rows_sorted))
best = rows_sorted[0]

lines = []
lines.append("Top configurations (sorted by fused_miou):")
for i, row in enumerate(rows_sorted[:top_k], start=1):
    lines.append(
        f"{i:2d}. fused_mIoU={fnum(row.get('fused_miou')):.2f} | "
        f"mode={row.get('crop_mode', '')} fill={row.get('outside_fill', '')} "
        f"coarse={row['coarse_n_segments']}/{row['coarse_min_size']} "
        f"fine={row['fine_n_segments']}/{row['fine_min_size']} "
        f"beta={row['beta']}"
    )

class_cols = [k for k in best.keys() if k.startswith("fused_iou_")]
class_cols = sorted(class_cols)
lines.append("")
lines.append("Best configuration details:")
lines.append(
    f"mode={best.get('crop_mode', '')}, fill={best.get('outside_fill', '')}, "
    f"small_region_ratio_thresh={best.get('small_region_ratio_thresh', '')}, "
    f"small_region_zoom={best.get('small_region_zoom', '')}"
)
lines.append(
    f"coarse={best['coarse_n_segments']}/{best['coarse_min_size']}, "
    f"fine={best['fine_n_segments']}/{best['fine_min_size']}, beta={best['beta']}"
)
lines.append(
    f"coarse_mIoU={fnum(best.get('coarse_miou')):.2f}, "
    f"fine_mIoU={fnum(best.get('fine_miou')):.2f}, "
    f"fused_mIoU={fnum(best.get('fused_miou')):.2f}"
)
if class_cols:
    lines.append("Per-class IoU (best fused):")
    for col in class_cols:
        class_name = col.replace("fused_iou_", "")
        lines.append(f"  {class_name}: {fnum(best.get(col)):.2f}")

text = "\n".join(lines)
print(text)
with open(summary_path, "w") as f:
    f.write(text + "\n")
print(f"\nWrote summary: {summary_path}")
PY

echo "Done."
