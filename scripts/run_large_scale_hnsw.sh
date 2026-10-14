#!/usr/bin/env bash
# HNSW version of run_large_scale_flat.sh (Ours-HNSW). Both code lookups use a
# Hamming HNSW graph; the gate, candidate augmentation, fusion, and exact
# reranking are unchanged.
# Usage: bash scripts/run_large_scale_hnsw.sh [extra args]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON:-python}"
DATASET="${H2MATS_DATASET:-beijing}"
DISTANCE_TYPE="${H2MATS_DISTANCE_TYPE:-TP}"
BENCHMARK="${H2MATS_BENCHMARK_DIR:-ann_artifacts/${DATASET}/${DISTANCE_TYPE}/augmented_1m}"
CODE_ASSETS="${H2MATS_CODE_ASSETS_DIR:-ann_artifacts/${DATASET}/${DISTANCE_TYPE}/candidate_selection/code_assets}"
QUERY_COUNT="${H2MATS_QUERY_COUNT:-2000}"
OUTPUT_DIR="${H2MATS_OUTPUT_DIR:-${BENCHMARK}/candidate_selection_hnsw}"
GALLERY_SIZES="${H2MATS_GALLERY_SIZES:-500000,600000,700000,800000,900000,1000000}"
CODE_DEVICE="${H2MATS_CODE_CUDA:-cuda:0}"
# Larger graph parameters keep the search robust for the many duplicate
# 128-bit codes at the 500K-1M scale.
HNSW_M="${H2MATS_CODE_HNSW_M:-32}"
HNSW_EF_CONSTRUCTION="${H2MATS_CODE_HNSW_EF_CONSTRUCTION:-400}"
HNSW_EF_SEARCH="${H2MATS_CODE_HNSW_EF_SEARCH:-2048}"

WEIGHT_ARGS=()
if [[ -n "${H2MATS_DUAL_METRIC_WEIGHT_FILE:-}" ]]; then
  WEIGHT_ARGS+=(--dual-metric-weight-file "${H2MATS_DUAL_METRIC_WEIGHT_FILE}")
fi

exec "${PYTHON_BIN}" candidate_selection.py \
  --gallery-file "${H2MATS_GALLERY_FILE:-${BENCHMARK}/gallery_embeddings.npz}" \
  --query-file "${H2MATS_QUERY_FILE:-${BENCHMARK}/query_embeddings.npz}" \
  --code-model-file "${CODE_ASSETS}/code_encoder_128_base.pt" \
  --metric-code-model-file "${CODE_ASSETS}/code_encoder_128_metric.pt" \
  "${WEIGHT_ARGS[@]}" \
  --hierarchy-dir "dataset/${DATASET}/hierarchy_louvain" \
  --output-dir "${OUTPUT_DIR}" \
  --gallery-sizes "${GALLERY_SIZES}" \
  --gallery-selection prefix \
  --positive-match-mode initial-only \
  --methods adaptive-dual-metric,adaptive-dual-metric-no-rerank,code-scan \
  --code-backend hnsw \
  --code-hnsw-m "${HNSW_M}" \
  --code-hnsw-ef-construction "${HNSW_EF_CONSTRUCTION}" \
  --code-hnsw-ef-search "${HNSW_EF_SEARCH}" \
  --max-queries "${QUERY_COUNT}" \
  --top-k 50 \
  --rerank-metric "${DISTANCE_TYPE}" \
  --tp-point-distance-file "ground_truth/${DATASET}/Point_dis_matrix.npy" \
  --code-device "${CODE_DEVICE}" \
  --query-code-device "${CODE_DEVICE}" \
  --hamming-candidates 768 \
  --adaptive-initial-global 768 \
  --adaptive-max-global 768 \
  --adaptive-initial-structural 96 \
  --adaptive-max-structural 192 \
  --min-route-candidates 96 \
  --max-route-candidates 192 \
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
  --metric-prefilter-candidates 64 \
  --metric-prefilter-high-quality-candidates 50 \
  --metric-prefilter-code-score soft \
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
  --dual-metric-code-weight 0.02 \
  --dual-metric-topology-weight 0.02 \
  --tp-eval \
  --tp-max-queries "${QUERY_COUNT}" \
  --tp-top-k 50 \
  --tp-reuse-max-distance-table \
  --tp-distance-table-file "${H2MATS_TP_DISTANCE_TABLE_FILE:-${BENCHMARK}/tp_distance_table_q${QUERY_COUNT}_g1000000.npy}" \
  "$@"
