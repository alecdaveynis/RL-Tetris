#!/usr/bin/env bash
# ~30 min improved-DQN continue from the current best checkpoint.
# Last 500-ep chunk (~15 min) peaked at ep 200 then declined; resume best, not latest.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

EPISODES="${1:-1000}"
RESUME="${2:-models/improved/dqn_best.pt}"

python -m tetris_rl.training.train_dqn \
  --preset improved \
  --episodes "$EPISODES" \
  --seed 42 \
  --resume "$RESUME" \
  --max-pieces 400 \
  --eval-every 100 \
  --eval-games 10 \
  --save-every 100 \
  --no-tb \
  --checkpoint-dir models/improved \
  --log-dir results/train_improved_30m_2
