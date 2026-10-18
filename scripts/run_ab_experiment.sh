#!/usr/bin/env bash
# Equal-budget A/B training: original vs improved DQN presets.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

EPISODES="${1:-400}"
SEED=42
MAX_PIECES=200

echo "=== Backup existing checkpoints ==="
mkdir -p models/backup_pre_audit
cp -n models/dqn_*.pt models/backup_pre_audit/ 2>/dev/null || true

echo "=== Config A: original (${EPISODES} eps) ==="
python -m tetris_rl.training.train_dqn \
  --preset original \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --max-pieces "$MAX_PIECES" \
  --eval-every 100 \
  --eval-games 5 \
  --save-every 100 \
  --no-tb \
  --checkpoint-dir models/original \
  --log-dir results/train_original

echo "=== Config B: improved (${EPISODES} eps) ==="
python -m tetris_rl.training.train_dqn \
  --preset improved \
  --episodes "$EPISODES" \
  --seed "$SEED" \
  --max-pieces "$MAX_PIECES" \
  --eval-every 100 \
  --eval-games 5 \
  --save-every 100 \
  --no-tb \
  --checkpoint-dir models/improved \
  --log-dir results/train_improved

echo "=== Evaluation (100 games, shared seeds) ==="
python -m tetris_rl.evaluation.compare_ab \
  --games 100 \
  --seed 0 \
  --max-pieces 500 \
  --original-model models/original/dqn_best.pt \
  --improved-model models/improved/dqn_best.pt \
  --out-dir results/ab_comparison

echo "Done."
