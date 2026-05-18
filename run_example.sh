#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python decontext.py \
  --model llava \
  --eval pope \
  --method mgap \
  --urs_alpha 0.75 \
  --gate_sensitivity 0.6 \
  --calibration_samples 50 \
  "$@"
