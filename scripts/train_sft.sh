#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
accelerate launch --config_file "${ACCELERATE_CONFIG:-configs/accelerate_zero2.yaml}" \
  training/train_vlm.py --config configs/sft.yaml "$@"
