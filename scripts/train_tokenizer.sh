#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
accelerate launch --config_file "${ACCELERATE_CONFIG:-configs/accelerate_ddp.yaml}" \
  training/train_tokenizer.py --config configs/tokenizer_g16.yaml "$@"
