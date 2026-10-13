#!/usr/bin/env bash
# End-to-end large-scale experiment: builds the augmented Query/Gallery
# benchmark, exports the embeddings, and evaluates candidate selection over
# growing gallery sizes. The hash heads come from run_candidate_selection.sh.
# Usage: bash scripts/run_large_scale_pipeline.sh <checkpoint> [extra args]
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON:-python}"
DATASET="${H2MATS_DATASET:-porto}"
DISTANCE_TYPE="${H2MATS_DISTANCE_TYPE:-TP}"
CHECKPOINT="${H2MATS_CHECKPOINT:-}"
if [[ $# -ge 1 && "${1}" != --* ]]; then CHECKPOINT="${1}"; shift; fi
if [[ -z "${CHECKPOINT}" ]]; then
  echo "Usage: $0 <checkpoint> [extra args]" >&2
  exit 1
fi

BENCHMARK_DIR="${H2MATS_BENCHMARK_DIR:-ann_artifacts/${DATASET}/${DISTANCE_TYPE}/augmented_1m}"
OUTPUT_DIR="${H2MATS_OUTPUT_DIR:-${BENCHMARK_DIR}/candidate_selection}"
HASH_ASSETS_DIR="${H2MATS_HASH_ASSETS_DIR:-ann_artifacts/${DATASET}/${DISTANCE_TYPE}/candidate_selection/hash_assets}"
CANDIDATE_HASH_MODEL="${H2MATS_CANDIDATE_HASH_MODEL:-${HASH_ASSETS_DIR}/hash_head_128_candidate.pt}"
METRIC_HASH_MODEL="${H2MATS_METRIC_HASH_MODEL:-${HASH_ASSETS_DIR}/hash_head_128_metric.pt}"
POINT_DISTANCE_FILE="${H2MATS_POINT_DISTANCE_FILE:-ground_truth/${DATASET}/Point_dis_matrix.npy}"
QUERY_COUNT="${H2MATS_QUERY_COUNT:-2000}"
TP_DISTANCE_TABLE_FILE="${H2MATS_TP_DISTANCE_TABLE_FILE:-${BENCHMARK_DIR}/tp_distance_table_q${QUERY_COUNT}_g1000000.npy}"
HIERARCHY_DIR="${H2MATS_HIERARCHY_DIR:-dataset/${DATASET}/hierarchy_louvain}"
GALLERY_SIZES="${H2MATS_GALLERY_SIZES:-500000,1000000}"
HASH_DEVICE="${H2MATS_HASH_CUDA:-cuda:0}"
MODEL_CUDA="${H2MATS_MODEL_CUDA:-0}"
TP_THREADS="${H2MATS_TP_THREADS:-64}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[error] required file not found: $1" >&2
    exit 2
  fi
}

require_file "${CHECKPOINT}"
require_file "${CANDIDATE_HASH_MODEL}"
require_file "${METRIC_HASH_MODEL}"
require_file "${POINT_DISTANCE_FILE}"
if [[ ! -d "${HIERARCHY_DIR}" ]]; then
  echo "[error] hierarchy directory not found: ${HIERARCHY_DIR}" >&2
  exit 2
fi
if [[ ! -f "${TP_DISTANCE_TABLE_FILE}" ]]; then
  echo "[info] the exact distance table will be computed and cached at ${TP_DISTANCE_TABLE_FILE}"
  echo "[info] a 2,000 x 1,000,000 float32 table needs about 8 GB of disk space."
fi

ARGS=(
  --dataset "${DATASET}"
  --distance-type "${DISTANCE_TYPE}"
  --benchmark-dir "${BENCHMARK_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --model "${CHECKPOINT}"
  --hash-model-file "${CANDIDATE_HASH_MODEL}"
  --metric-hash-model-file "${METRIC_HASH_MODEL}"
  --hierarchy-dir "${HIERARCHY_DIR}"
  --tp-point-distance-file "${POINT_DISTANCE_FILE}"
  --tp-distance-table-file "${TP_DISTANCE_TABLE_FILE}"
  --gallery-sizes "${GALLERY_SIZES}"
  --query-count "${QUERY_COUNT}"
  --top-k 50
  --tp-top-k 50
  --topology-method adaptive-dual-metric
  --rerank-metric "${DISTANCE_TYPE}"
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
  --hamming-candidates 768
  --hash-backend "${H2MATS_HASH_BACKEND:-mih}"
  --hash-flat-block-size "${H2MATS_HASH_FLAT_BLOCK_SIZE:-65536}"
  --adaptive-max-rerank 960
  --dual-metric-search-candidates 256
  --dual-metric-candidates 64
  --dual-metric-high-quality-tp-candidates 64
  --dual-metric-tp-candidates 128
  --dual-metric-hard-tp-candidates 160
  --dual-metric-hard-margin-fraction 0.50
  --dual-metric-head-protect 48
  --dual-metric-structural-protect 16
  --dual-metric-embedding-weight 0.96
  --dual-metric-hash-weight 0.02
  --dual-metric-topology-weight 0.02
  --metric-prefilter-hash-score soft
  --adaptive-preserve-global-top50
  --include-global-hash
  --skip-hash-training
  --skip-hnsw
  --hash-device "${HASH_DEVICE}"
  --topology-query-code-device "${HASH_DEVICE}"
  --cuda "${MODEL_CUDA}"
  --tp-num-threads "${TP_THREADS}"
)

if [[ "${H2MATS_SKIP_EXPORT:-0}" == "1" ]]; then
  require_file "${OUTPUT_DIR}/our_gallery_embeddings.npz"
  require_file "${OUTPUT_DIR}/our_query_embeddings.npz"
  ARGS+=(--skip-export)
fi
if [[ "${H2MATS_DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

exec "${PYTHON_BIN}" run_augmented_tp_experiment.py "${ARGS[@]}" "$@"
