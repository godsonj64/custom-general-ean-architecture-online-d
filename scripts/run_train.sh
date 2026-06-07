#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_train.sh  —  Launch EAN training
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${REPO_ROOT}/configs/default.yaml"
OUTPUT_DIR="${REPO_ROOT}/outputs"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${REPO_ROOT}/data"

echo "======================================="
echo " EAN Training"
echo "  Config : ${CONFIG}"
echo "  Output : ${OUTPUT_DIR}"
echo "======================================="

# Optional: resume from a checkpoint if one exists
RESUME_ARG=""
if [ -f "${OUTPUT_DIR}/last_model.pt" ]; then
    echo "Found existing checkpoint — resuming training."
    RESUME_ARG="--resume ${OUTPUT_DIR}/last_model.pt"
fi

python "${REPO_ROOT}/src/train.py" \
    --config "${CONFIG}" \
    ${RESUME_ARG}

echo "======================================="
echo " Training complete!"
echo " Checkpoints : ${OUTPUT_DIR}/"
echo "======================================="
