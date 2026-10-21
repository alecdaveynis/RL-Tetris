#!/usr/bin/env bash
# Controlled LR smoke from an identical frozen best checkpoint.
# Does not write into models/improved/.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

START="${START:-models/frozen/dqn_best_val1320.pt}"
EPISODES="${1:-40}"
DEVICE="${DEVICE:-}"

DEV_ARGS=()
if [[ -n "${DEVICE}" ]]; then
  DEV_ARGS=(--device "$DEVICE")
fi

echo "=== LR 5e-4 smoke ==="
python -m tetris_rl.training.train_dqn \
  --preset improved \
  --episodes "$EPISODES" \
  --seed 42 \
  --resume "$START" \
  --lr 0.0005 \
  --max-pieces 400 \
  --eval-every 20 \
  --eval-games 10 \
  --save-every 20 \
  --no-tb \
  --checkpoint-dir models/exp_lr_5e4 \
  --log-dir results/exp_lr_5e4 \
  "${DEV_ARGS[@]+"${DEV_ARGS[@]}"}"

echo "=== LR 1e-4 smoke ==="
python -m tetris_rl.training.train_dqn \
  --preset improved \
  --episodes "$EPISODES" \
  --seed 42 \
  --resume "$START" \
  --lr 0.0001 \
  --max-pieces 400 \
  --eval-every 20 \
  --eval-games 10 \
  --save-every 20 \
  --no-tb \
  --checkpoint-dir models/exp_lr_1e4 \
  --log-dir results/exp_lr_1e4 \
  "${DEV_ARGS[@]+"${DEV_ARGS[@]}"}"
