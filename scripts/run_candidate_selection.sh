#!/usr/bin/env bash
# Hierarchy-aware adaptive candidate selection on the standard test set.
# The first run exports the embeddings and trains the base and metric hash
# heads into ${OUTPUT_DIR}/hash_assets; later runs reuse them unless
# --retrain-hash is given.
# Usage: bash scripts/run_candidate_selection.sh <checkpoint> [output-dir] [extra args]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON:-python}"
DATASET="${H2MATS_DATASET:-porto}"
DISTANCE_TYPE="${H2MATS_DISTANCE_TYPE:-TP}"
CHECKPOINT="${H2MATS_CHECKPOINT:-}"
OUTPUT_DIR="${H2MATS_OUTPUT_DIR:-ann_artifacts/${DATASET}/${DISTANCE_TYPE}/candidate_selection}"
POINT_DISTANCE_FILE="${H2MATS_POINT_DISTANCE_FILE:-ground_truth/${DATASET}/Point_dis_matrix.npy}"
HASH_DEVICE="${H2MATS_HASH_CUDA:-cuda:0}"
MODEL_CUDA="${H2MATS_MODEL_CUDA:-0}"
TP_THREADS="${H2MATS_TP_THREADS:-64}"

if [[ $# -ge 1 && "${1}" != --* ]]; then CHECKPOINT="${1}"; shift; fi
if [[ $# -ge 1 && "${1}" != --* ]]; then OUTPUT_DIR="${1}"; shift; fi
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: $0 <checkpoint> [output-dir] [extra args]" >&2
  exit 1
fi

ASSETS_DIR="${H2MATS_ASSETS_DIR:-${OUTPUT_DIR}/hash_assets}"
REUSE_ARGS=()
if [[ -f "${ASSETS_DIR}/hash_head_128_candidate.pt" ]]; then
  REUSE_ARGS+=(--reuse-hash-assets-dir "${ASSETS_DIR}")
fi

exec "${PYTHON_BIN}" run_adaptive_topo_hash_checkpoint.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --distance-type "${DISTANCE_TYPE}" \
  --output-dir "${OUTPUT_DIR}" \
  "${REUSE_ARGS[@]}" \
  --bits 128 \
  --cuda "${MODEL_CUDA}" \
  --hash-device "${HASH_DEVICE}" \
  --query-code-device "${HASH_DEVICE}" \
  --dual-hash-heads \
  --methods adaptive-dual-metric,adaptive-dual-metric-no-rerank,global-hash \
  --adaptive-initial-global 768 \
  --adaptive-max-global 768 \
  --adaptive-initial-structural 96 \
  --adaptive-max-structural 192 \
  --min-route-candidates 96 \
  --max-route-candidates 192 \
  --hamming-candidates 768 \
  --global-candidates 768 \
  --structural-candidates 192 \
  --max-rerank-candidates 960 \
  --adaptive-max-rerank 960 \
  --adaptive-min-route-recall 48 \
  --adaptive-gate hybrid \
  --adaptive-global-margin 0.02 \
  --adaptive-metric-fast-path skip \
  --adaptive-max-tp-fraction 0.1 \
  --adaptive-preserve-global-top50 \
  --rerank-metric "${DISTANCE_TYPE}" \
  --point-distance-file "${POINT_DISTANCE_FILE}" \
  --metric-prefilter-hash-score soft \
  --metric-prefilter-soft-temperature 1.0 \
  --dual-metric-search-candidates 256 \
  --dual-metric-candidates 64 \
  --dual-metric-high-quality-tp-candidates 64 \
  --dual-metric-tp-candidates 128 \
  --dual-metric-hard-tp-candidates 160 \
  --dual-metric-hard-margin-fraction 0.50 \
  --dual-metric-head-protect 48 \
  --dual-metric-structural-protect 16 \
  --dual-metric-embedding-weight 0.96 \
  --dual-metric-hash-weight 0.02 \
  --dual-metric-topology-weight 0.02 \
  --tp-num-threads "${TP_THREADS}" \
  "$@"
