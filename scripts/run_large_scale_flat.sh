#!/usr/bin/env bash
# Large-scale candidate selection on exported Query/Gallery embeddings with a
# CPU full-gallery XOR/popcount code scan.
# Usage: bash scripts/run_large_scale_flat.sh [extra args]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON:-python}"
DATASET="${H2MATS_DATASET:-porto}"
DISTANCE_TYPE="${H2MATS_DISTANCE_TYPE:-TP}"
BENCHMARK="${H2MATS_BENCHMARK_DIR:-ann_artifacts/${DATASET}/${DISTANCE_TYPE}/augmented_1m}"
CODE_ASSETS="${H2MATS_CODE_ASSETS_DIR:-ann_artifacts/${DATASET}/${DISTANCE_TYPE}/candidate_selection/code_assets}"
GALLERY_FILE="${H2MATS_GALLERY_FILE:-${BENCHMARK}/gallery_embeddings.npz}"
QUERY_FILE="${H2MATS_QUERY_FILE:-${BENCHMARK}/query_embeddings.npz}"
BASE_ENCODER="${H2MATS_BASE_CODE_MODEL:-${CODE_ASSETS}/code_encoder_128_base.pt}"
METRIC_ENCODER="${H2MATS_METRIC_CODE_MODEL:-${CODE_ASSETS}/code_encoder_128_metric.pt}"
HIERARCHY="${H2MATS_HIERARCHY_DIR:-dataset/${DATASET}/hierarchy_louvain}"
POINT_DISTANCES="${H2MATS_POINT_DISTANCE_FILE:-ground_truth/${DATASET}/Point_dis_matrix.npy}"
QUERY_COUNT="${H2MATS_QUERY_COUNT:-2000}"
TP_TABLE="${H2MATS_TP_DISTANCE_TABLE_FILE:-${BENCHMARK}/tp_distance_table_q${QUERY_COUNT}_g1000000.npy}"
OUTPUT_DIR="${H2MATS_OUTPUT_DIR:-${BENCHMARK}/candidate_selection_flat}"
CODE_DEVICE="${H2MATS_CODE_CUDA:-cuda:0}"

ARGS=(
  --gallery-file "${GALLERY_FILE}"
  --query-file "${QUERY_FILE}"
  --code-model-file "${BASE_ENCODER}"
  --metric-code-model-file "${METRIC_ENCODER}"
  --hierarchy-dir "${HIERARCHY}"
  --output-dir "${OUTPUT_DIR}"
  --gallery-sizes "${H2MATS_GALLERY_SIZES:-500000,600000,700000,800000,900000,1000000}"
  --gallery-selection prefix
  --positive-match-mode initial-only
  --positives-per-query 1
  --methods adaptive-dual-metric,adaptive-dual-metric-no-rerank,code-scan
  --max-queries "${QUERY_COUNT}"
  --top-k 50
  --metric l2
  --rerank-metric "${DISTANCE_TYPE}"
  --code-backend flat
  --code-flat-block-size "${H2MATS_CODE_FLAT_BLOCK_SIZE:-65536}"
  --adaptive-gate hybrid
  --adaptive-global-margin 0.02
  --adaptive-metric-fast-path skip
  --adaptive-max-tp-fraction 0.10
  --adaptive-initial-global 768
  --adaptive-max-global 768
  --adaptive-initial-structural 96
  --adaptive-max-structural 192
  --adaptive-min-route-recall 48
  --min-route-candidates 96
  --max-route-candidates 192
  --structural-candidates 192
  --global-candidates 768
  --max-rerank-candidates 960
  --adaptive-max-rerank 960
  --hamming-candidates 768
  --dual-metric-search-candidates 256
  --dual-metric-candidates 64
  --dual-metric-high-quality-tp-candidates 64
  --dual-metric-tp-candidates 128
  --dual-metric-hard-tp-candidates 160
  --dual-metric-hard-margin-fraction 0.50
  --dual-metric-head-protect 48
  --dual-metric-structural-protect 16
  --dual-metric-embedding-weight 0.96
  --dual-metric-code-weight 0.02
  --dual-metric-topology-weight 0.02
  --metric-prefilter-code-score soft
  --adaptive-preserve-global-top50
  --code-device "${CODE_DEVICE}"
  --query-code-device "${CODE_DEVICE}"
  --tp-eval
  --tp-point-distance-file "${POINT_DISTANCES}"
  --tp-max-queries "${QUERY_COUNT}"
  --tp-top-k 50
  --tp-num-threads "${H2MATS_TP_THREADS:-64}"
  --tp-reuse-max-distance-table
  --tp-distance-table-file "${TP_TABLE}"
)
ARGS+=("$@")

if [[ "${H2MATS_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${PYTHON_BIN}" candidate_selection.py "${ARGS[@]}"
  printf '\n'
  exit 0
fi

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] required file not found: $1" >&2
    exit 2
  fi
}

for file in "${GALLERY_FILE}" "${QUERY_FILE}" "${BASE_ENCODER}" "${METRIC_ENCODER}" "${POINT_DISTANCES}"; do
  require_file "${file}"
done
for name in node_to_l1.npy node_to_l2.npy node_to_l3.npy; do
  require_file "${HIERARCHY}/${name}"
done

exec "${PYTHON_BIN}" candidate_selection.py "${ARGS[@]}"
