#!/usr/bin/env bash
# Emb-HNSW baseline: HNSW search over exported continuous embeddings, without
# hashing or exact reranking. The exact distances serve only as ground truth.
# Usage: bash scripts/run_large_scale_emb_hnsw.sh [extra args]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON:-python}"
DATASET="${H2MATS_DATASET:-porto}"
DISTANCE_TYPE="${H2MATS_DISTANCE_TYPE:-TP}"
BENCHMARK="${H2MATS_BENCHMARK_DIR:-ann_artifacts/${DATASET}/${DISTANCE_TYPE}/augmented_1m}"
GALLERY_FILE="${H2MATS_GALLERY_FILE:-${BENCHMARK}/gallery_embeddings.npz}"
QUERY_FILE="${H2MATS_QUERY_FILE:-${BENCHMARK}/query_embeddings.npz}"
OUTPUT_DIR="${H2MATS_OUTPUT_DIR:-${BENCHMARK}/emb_hnsw}"
POINT_DISTANCES="${H2MATS_POINT_DISTANCE_FILE:-ground_truth/${DATASET}/Point_dis_matrix.npy}"
QUERY_COUNT="${H2MATS_QUERY_COUNT:-2000}"
TP_TABLE="${H2MATS_TP_DISTANCE_TABLE_FILE:-${BENCHMARK}/tp_distance_table_q${QUERY_COUNT}_g1000000.npy}"
GALLERY_SIZES="${H2MATS_GALLERY_SIZES:-500000,600000,700000,800000,900000,1000000}"
TOP_K="${H2MATS_TOP_K:-50}"
HNSW_M="${H2MATS_HNSW_M:-16}"
HNSW_EF_CONSTRUCTION="${H2MATS_HNSW_EF_CONSTRUCTION:-64}"
HNSW_EF_SEARCH="${H2MATS_HNSW_EF_SEARCH:-1024}"
HNSW_NUM_THREADS="${H2MATS_HNSW_NUM_THREADS:-1}"
TP_THREADS="${H2MATS_TP_THREADS:-64}"

ARGS=(
  --gallery-file "${GALLERY_FILE}"
  --query-file "${QUERY_FILE}"
  --output-dir "${OUTPUT_DIR}"
  --gallery-sizes "${GALLERY_SIZES}"
  --gallery-selection prefix
  --positive-match-mode initial-only
  --positives-per-query 1
  --methods hnsw
  --max-queries "${QUERY_COUNT}"
  --top-k "${TOP_K}"
  --metric l2
  --hnsw-m "${HNSW_M}"
  --hnsw-ef-construction "${HNSW_EF_CONSTRUCTION}"
  --hnsw-ef-search "${HNSW_EF_SEARCH}"
  --hnsw-num-threads "${HNSW_NUM_THREADS}"
  --tp-eval
  --tp-point-distance-file "${POINT_DISTANCES}"
  --tp-max-queries "${QUERY_COUNT}"
  --tp-top-k "${TOP_K}"
  --tp-num-threads "${TP_THREADS}"
  --tp-reuse-max-distance-table
  --tp-distance-table-file "${TP_TABLE}"
)
ARGS+=("$@")

if [[ "${H2MATS_DRY_RUN:-0}" == "1" ]]; then
  printf '%q ' "${PYTHON_BIN}" hnsw_large_scale.py "${ARGS[@]}"
  printf '\n'
  exit 0
fi

for file in "${GALLERY_FILE}" "${QUERY_FILE}" "${POINT_DISTANCES}"; do
  if [[ ! -f "${file}" ]]; then
    echo "[error] required file not found: ${file}" >&2
    exit 2
  fi
done

exec "${PYTHON_BIN}" hnsw_large_scale.py "${ARGS[@]}"
