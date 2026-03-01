#!/usr/bin/env bash
#
# Minimal GPU smoke test for ExMed-BERT pre-training (no streaming).
# Run from repo root. Uses sample .pt data and runs 5 steps.
#
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Optional: disable debug flags for faster run
# export CUDA_LAUNCH_BLOCKING=0
# export TORCH_USE_CUDA_DSA=0

DATASET_DIR="${DATASET_DIR:-pretrain_data}"
TRAIN_PT="${TRAIN_PT:-$DATASET_DIR/komodo_panta_sample_train.pt}"
VAL_PT="${VAL_PT:-$DATASET_DIR/komodo_panta_sample_val.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-output/minimal_test_pretrain}"
OUTPUT_DATA_DIR="${OUTPUT_DATA_DIR:-output/minimal_test_pretrain_data}"

if [[ ! -f "$TRAIN_PT" ]]; then
  echo "Missing training data: $TRAIN_PT"
  echo "Set TRAIN_PT and VAL_PT or place sample .pt files in pretrain_data/."
  exit 1
fi

echo "=========================================="
echo "ExMed-BERT minimal GPU smoke test"
echo "=========================================="
echo "  Train: $TRAIN_PT"
echo "  Val:   $VAL_PT"
echo "  Out:   $OUTPUT_DIR"
echo "  Steps: 5"
echo "=========================================="

python scripts/pretrain-exmed-bert-clinvec.py \
  "$TRAIN_PT" \
  "$VAL_PT" \
  "$OUTPUT_DIR" \
  "$OUTPUT_DATA_DIR" \
  --train-batch-size 8 \
  --eval-batch-size 4 \
  --max-steps 5 \
  --num-attention-heads 6 \
  --num-hidden-layers 6 \
  --hidden-size 288 \
  --intermediate-size 512 \
  --epochs 1 \
  --max-seq-length 512 \
  --logging-steps 2 \
  --eval-steps 5 \
  --save-steps 5 \
  --dynamic-masking

echo "=========================================="
echo "Minimal test finished. Check $OUTPUT_DIR"
echo "=========================================="
