#!/bin/bash
# Script to train ExMed-BERT on synthetic data
# This is a complete end-to-end pipeline

set -e  # Exit on error

echo "=========================================="
echo "ExMed-BERT Training on Synthetic Data"
echo "=========================================="

# Configuration

# Configuration
N_PATIENTS=${1:-1000}
NOVEL_PROB=${2:-0.01}
MAX_LENGTH=500
SEED=${4:-42}

# Paths
SYNTHETIC_JSON="data/komodo_sample_3000_converted.json"
DATASET_DIR="pretrain_stuff"
TRAIN_DATASET="${DATASET_DIR}/komodo_sample_3000_train.pt"
VAL_DATASET="${DATASET_DIR}/komodo_sample_3000_val.pt"
OUTPUT_DIR="output/komodo_sample_3000_pretrain"
OUTPUT_DATA_DIR="output/komodo_sample_3000_pretrain_data"

export PYTHONPATH=$(pwd):$PYTHONPATH


#python scripts/generate_synthetic_patients.py \
#    -n ${N_PATIENTS} \
#    -o ${SYNTHETIC_JSON} \
#    -p ${NOVEL_PROB} \
#    -s ${SEED}

# Step 2: Convert to PatientDataset format
echo ""
echo "Step 2: Converting JSON to PatientDataset format..."
echo "  Max sequence length: ${MAX_LENGTH}"
echo "  Train/val split: 80/20"
echo ""

python scripts/convert_synthetic_to_dataset.py \
    --input ${SYNTHETIC_JSON} \
    --output ${DATASET_DIR}/komodo_sample_3000.pt \
    --split all \
    --max-length 500 \
    --train-ratio 0.8 \
    --seed ${SEED}

