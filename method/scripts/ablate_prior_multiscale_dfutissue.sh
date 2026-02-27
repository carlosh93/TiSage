#!/usr/bin/env bash
# Ablation over coarse/fine superpixel params and fusion beta for MedSigLIP multi-scale prior.
# Uses a single-scale trained classifier. Writes results to a TSV and prints a LaTeX table.
#
# Usage (from repo root):
#   bash method/scripts/ablate_prior_multiscale_dfutissue.sh
#   CLASSIFIER_PATH=/path/to/medsiglip_head_dfutissue.pt bash method/scripts/ablate_prior_multiscale_dfutissue.sh
#
# Requires: Python with `scripts/eval_prior_multiscale_dfutissue.py`.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$METHOD_DIR/.." && pwd)"
cd "$METHOD_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CLASSIFIER_PATH="${CLASSIFIER_PATH:-$METHOD_DIR/checkpoints/pretrained/medsiglip_head_dfutissue.pt}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/DFUTissue}"
VAL_TXT="${VAL_TXT:-$PROJECT_ROOT/splits/dfutissue/val.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results_medsiglip_multiscale_ablation}"
TSV_PATH="$OUTPUT_DIR/ablation_results.tsv"

if [[ ! -f "$CLASSIFIER_PATH" ]]; then
  echo "ERROR: Classifier not found at $CLASSIFIER_PATH. Set CLASSIFIER_PATH or train and save one first."
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
# Start fresh TSV (ablation script appends rows)
rm -f "$TSV_PATH"

# Grid: coarse_n_segments coarse_min_size fine_n_segments fine_min_size beta
coarse_n_arr=(24 32 64)
coarse_min_arr=(80 100)
fine_n_arr=(96 128)
fine_min_arr=(40 50)
beta_arr=(0.3 0.5 0.7)

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

echo "Running $total ablation runs (classifier: $CLASSIFIER_PATH). Results -> $TSV_PATH"
n=0
for coarse_n in "${coarse_n_arr[@]}"; do
  for coarse_min in "${coarse_min_arr[@]}"; do
    for fine_n in "${fine_n_arr[@]}"; do
      for fine_min in "${fine_min_arr[@]}"; do
        for beta in "${beta_arr[@]}"; do
          n=$((n + 1))
          echo "[$n/$total] coarse=${coarse_n}/${coarse_min} fine=${fine_n}/${fine_min} beta=$beta"
          "$PYTHON_BIN" scripts/eval_prior_multiscale_dfutissue.py \
            --classifier-path "$CLASSIFIER_PATH" \
            --data-root "$DATA_ROOT" \
            --val-txt "$VAL_TXT" \
            --coarse-n-segments "$coarse_n" \
            --coarse-min-size "$coarse_min" \
            --fine-n-segments "$fine_n" \
            --fine-min-size "$fine_min" \
            --beta "$beta" \
            --results-tsv "$TSV_PATH" \
            --quiet
        done
      done
    done
  done
done

echo ""
echo "Generating LaTeX table..."
"$PYTHON_BIN" - "$TSV_PATH" "$OUTPUT_DIR/ablation_table.tex" << 'PY'
import sys
import csv

tsv_path = sys.argv[1]
out_path = sys.argv[2]

with open(tsv_path, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

if not rows:
    print("No rows in TSV.")
    sys.exit(0)

# Find best by fused_miou
best_idx = 0
best_miou = -1.0
for i, r in enumerate(rows):
    try:
        v = float(r.get("fused_miou", -1))
        if v > best_miou:
            best_miou = v
            best_idx = i
    except (ValueError, TypeError):
        pass

def num(s, fmt="{:.2f}"):
    try:
        return fmt.format(float(s))
    except (ValueError, TypeError):
        return str(s)

# Build LaTeX table: Config (short) | Coarse mIoU | Fine mIoU | Fused mIoU | Bg | Fibrin | Gran | Callus
lines = []
lines.append(r"% Requires \usepackage{booktabs} in preamble.")
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\caption{MedSigLIP multi-scale prior ablation (single-scale trained classifier). Val.\ mIoU (\%) and per-class IoU.}")
lines.append(r"\label{tab:medsiglip-multiscale-ablation}")
lines.append(r"\begin{tabular}{lccccccc}")
lines.append(r"\toprule")
lines.append(r"Config & Coarse & Fine & Fused & Bg & Fibrin & Gran & Callus \\")
lines.append(r"\midrule")

for i, r in enumerate(rows):
    cfg = f"c{r['coarse_n_segments']}/{r['coarse_min_size']} f{r['fine_n_segments']}/{r['fine_min_size']} $\\beta$={r['beta']}"
    coarse = num(r.get("coarse_miou"))
    fine = num(r.get("fine_miou"))
    fused = num(r.get("fused_miou"))
    bg = num(r.get("fused_iou_background", ""))
    fib = num(r.get("fused_iou_fibrin", ""))
    gran = num(r.get("fused_iou_granulation", ""))
    call = num(r.get("fused_iou_callus", ""))
    fused_cell = f"\\textbf{{{fused}}}" if i == best_idx else fused
    lines.append(f"{cfg} & {coarse} & {fine} & {fused_cell} & {bg} & {fib} & {gran} & {call} \\\\")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\end{table}")

with open(out_path, "w") as f:
    f.write("\n".join(lines))

print(f"Best config (row {best_idx + 1}): fused mIoU = {best_miou:.2f}%")
print(f"  coarse_n_segments={rows[best_idx].get('coarse_n_segments')} coarse_min_size={rows[best_idx].get('coarse_min_size')} fine_n_segments={rows[best_idx].get('fine_n_segments')} fine_min_size={rows[best_idx].get('fine_min_size')} beta={rows[best_idx].get('beta')}")
print(f"LaTeX table written to {out_path}")
print("")
print("--- LaTeX (paste into paper) ---")
print("\n".join(lines))
PY

echo "Done. TSV: $TSV_PATH  LaTeX: $OUTPUT_DIR/ablation_table.tex"
