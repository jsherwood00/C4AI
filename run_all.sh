#!/usr/bin/env bash
#
# run_all.sh — one-shot regeneration of every paper figure.
#
# Pipeline:
#   1. Compute Bradley-Terry ratings from results.csv → bt_ratings.json
#   2. Run the BT-derived figure scripts that consume bt_ratings.json
#   3. Run the input-derived figure scripts that consume results.csv,
#      player_times_seconds.txt, or transcripts/
#
# Outputs land in outputs/pdfs/. The intermediate transcript_analysis.csv
# is written to outputs/.
#
# Usage:
#     ./run_all.sh

set -euo pipefail

INPUT_DIR="${INPUT_DIR:-input}"
SCRIPTS_DIR="${SCRIPTS_DIR:-scripts}"
OUTPUTS_DIR="${OUTPUTS_DIR:-outputs}"

PDFS_DIR="${OUTPUTS_DIR}/pdfs"
RATINGS_JSON="${OUTPUTS_DIR}/bt_ratings.json"
TRANSCRIPTS_CSV="${OUTPUTS_DIR}/transcript_analysis.csv"

RESULTS_CSV="${INPUT_DIR}/results.csv"
TIMES_TXT="${INPUT_DIR}/player_times_seconds.txt"
TRANSCRIPTS_DIR="${INPUT_DIR}/transcripts"

mkdir -p "${PDFS_DIR}"

# -----------------------------------------------------------------------------
# Step 1 — Compute BT ratings
# -----------------------------------------------------------------------------
echo "[1/4] Computing BT ratings..."
python3 "${SCRIPTS_DIR}/bt.py" \
    --results "${RESULTS_CSV}" \
    --output  "${RATINGS_JSON}"

# -----------------------------------------------------------------------------
# Step 2 — BT-derived figures (read bt_ratings.json)
# -----------------------------------------------------------------------------
echo "[2/4] BT-derived figures..."
python3 "${SCRIPTS_DIR}/best_bt_lollipop.py"   --ratings "${RATINGS_JSON}"  --out "${PDFS_DIR}/best_bt_lollipop.pdf"
python3 "${SCRIPTS_DIR}/probe_bt.py"           --ratings "${RATINGS_JSON}"  --out "${PDFS_DIR}/probe_bt.pdf"
python3 "${SCRIPTS_DIR}/main_bt.py"            --ratings "${RATINGS_JSON}"  --out "${PDFS_DIR}/main_bt.pdf"
python3 "${SCRIPTS_DIR}/mean_bt_vs_release.py" --ratings "${RATINGS_JSON}"  --out "${PDFS_DIR}/mean_bt_vs_release.pdf"

# -----------------------------------------------------------------------------
# Step 3 — Input-derived figures (read results.csv / player_times_seconds.txt)
# -----------------------------------------------------------------------------
echo "[3/4] Input-derived figures..."
python3 "${SCRIPTS_DIR}/move_time_bt.py"   --results "${RESULTS_CSV}" --out "${PDFS_DIR}/move_time_bt.pdf"

# First-mover, wins-only variants only
python3 "${SCRIPTS_DIR}/first_mover_optimal.py" \
    --results "${RESULTS_CSV}" --plot bars --success-type wins-only \
    --out "${PDFS_DIR}/first_mover_optimal_bars_wins_only.pdf"
python3 "${SCRIPTS_DIR}/first_mover_optimal.py" \
    --results "${RESULTS_CSV}" --plot release --success-type wins-only \
    --out "${PDFS_DIR}/first_mover_optimal_release_wins_only.pdf"

# Time-budget figures: 3-group (main / eval / non-eval pooled) and the new
# 4-group (Opus 4.7 / Opus 4.6 / Gemini / GPT-5.4 main).
python3 "${SCRIPTS_DIR}/budget_3group.py" --times "${TIMES_TXT}" --out "${PDFS_DIR}/budget_3group.pdf"
python3 "${SCRIPTS_DIR}/budget_4main.py"  --times "${TIMES_TXT}" --out "${PDFS_DIR}/budget_4main.pdf"

# -----------------------------------------------------------------------------
# Step 4 — Transcript-derived figures (keyword heatmap + eval-vs-noneval top4)
# -----------------------------------------------------------------------------
echo "[4/4] Transcript-derived figures..."
python3 "${SCRIPTS_DIR}/transcript_analysis.py" "${TRANSCRIPTS_DIR}" \
    -p output.txt -o "${TRANSCRIPTS_CSV}"

# plot_heatmap.py writes <basename>.pdf (no PNG/SVG anymore).
python3 "${SCRIPTS_DIR}/plot_heatmap.py" "${TRANSCRIPTS_CSV}" \
    -o "${PDFS_DIR}/keyword_heatmap"

python3 "${SCRIPTS_DIR}/plot_top4.py" "${TRANSCRIPTS_CSV}" \
    --out "${PDFS_DIR}/eval_vs_noneval_top4.pdf"

echo
echo "Done. ${PDFS_DIR}/ contents:"
ls -1 "${PDFS_DIR}"
