#!/usr/bin/env bash
# Short smoke training run to verify the pipeline end-to-end.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m tetris_rl.training.train_dqn \
  --episodes 30 \
  --seed 42 \
  --min-buffer-size 64 \
  --batch-size 32 \
  --epsilon-decay-steps 200 \
  --eval-every 15 \
  --eval-games 3 \
  --save-every 15 \
  --max-pieces 100 \
  --no-tb \
  --checkpoint-dir models \
  --log-dir results/train_logs
