#!/usr/bin/env bash
# Evaluate the embedding-based retrieval accuracy of a trained checkpoint.
# Usage: bash scripts/eval_embedding.sh <checkpoint-directory-or-epoch-file>
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON:-python}"
DATASET="${H2MATS_DATASET:-porto}"
DISTANCE_TYPE="${H2MATS_DISTANCE_TYPE:-TP}"
CUDA_ID="${H2MATS_MODEL_CUDA:-0}"
CHECKPOINT="${1:-${H2MATS_CHECKPOINT:-}}"

if [[ -z "${CHECKPOINT}" || ! -e "${CHECKPOINT}" ]]; then
  echo "Usage: $0 <checkpoint-directory-or-epoch-file>" >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" eval_spatial_model.py \
  --checkpoint "${CHECKPOINT}" \
  --dataset "${DATASET}" \
  --distance-type "${DISTANCE_TYPE}" \
  --use-hybrid-geo \
  --graph-encoder hybrid_multilevel \
  --seq-encoder hmt \
  --coord-encoder transformer \
  --hmt-backbone hybrid \
  --hmt-layers 2 \
  --hmt-kernel-size 5 \
  --hmt-dropout 0.1 \
  --hmt-transformer-heads 4 \
  --hmt-scale-aligned-graph \
  --coord-fusion gated_attention \
  --cuda "${CUDA_ID}"
