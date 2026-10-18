# Tetris-RL

Deep reinforcement learning for Tetris 

A self-contained Python project that implements a Tetris game engine, a Gymnasium placement-level environment, heuristic and random baselines, and a custom PyTorch Deep Q-Network that learns to choose piece placements. Includes reproducible training, evaluation, comparison charts, and a Pygame UI for human play and AI demos.

## Project overview

Agents act at the **placement level**: for each current piece they choose a legal hard-drop (rotation + column), not frame-by-frame keyboard moves. The DQN scores candidate placements from state-action features of the resulting board. Training uses shaped rewards; evaluation reports true game score and lines cleared.

**Why not Stable-Baselines3 DQN?** SB3’s standard DQN expects a fixed discrete action space and does **not** support action masking out of the box. Placement Tetris has a variable legal set each turn, so this project implements a genuine placement-action Q-network in PyTorch. SB3 remains in `requirements.txt` for optional future work (e.g. PPO).

## Architecture

```mermaid
flowchart TB
    subgraph Engine["tetris_rl/engine.py"]
        Board[Board 10x20]
        Pieces[7-bag Tetrominoes]
        Place[Placement enumeration]
    end

    subgraph Env["tetris_rl/environment.py"]
        Obs[Observation vector]
        Mask[Legal action mask]
        Rew[Shaped rewards]
    end

    subgraph Agents["Agents"]
        Rand[Random]
        Heur[Heuristic]
        DQN[Placement DQN]
    end

    subgraph Train["Training"]
        Buf[Replay buffer]
        Opt[Bellman / Huber loss]
        CKPT[Checkpoints]
    end

    subgraph Eval["Evaluation"]
        Metrics[Score / lines / survival]
        Charts[Matplotlib charts]
    end

    UI[Pygame UI]

    Engine --> Env
    Env --> Agents
    DQN --> Buf --> Opt --> CKPT
    Agents --> Eval --> Charts
    Engine --> UI
    Agents --> UI
```

| Path | Role |
|------|------|
| `tetris_rl/engine.py` | Pure game logic (no RL / no rendering) |
| `tetris_rl/features.py` | Board metrics and observation / state-action features |
| `tetris_rl/rewards.py` | Configurable shaped training rewards |
| `tetris_rl/environment.py` | Gymnasium env + legal-action mask |
| `tetris_rl/agents/` | Random, heuristic, DQN |
| `tetris_rl/training/` | Replay buffer + training CLI |
| `tetris_rl/evaluation/` | Seeded evaluation and comparison |
| `tetris_rl/ui/pygame_app.py` | Human play + AI watch mode |
| `tests/` | pytest suite |
| `models/` | Checkpoints |
| `results/` | Logs, CSVs, charts |

## Installation

Requires **Python 3.11+** (3.12–3.14 also work). The UI depends on `pygame-ce` (API-compatible with pygame; preferred because it provides wheels for newer Python versions).

```bash
cd RL-Tetris
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## How to play manually

```bash
export PYTHONPATH=.
python -m tetris_rl.ui.pygame_app --mode human
```

| Key | Action |
|-----|--------|
| ← / → | Move |
| ↑ | Rotate |
| Space | Hard drop |
| 1 / 2 / 3 / 4 | Human / Random / Heuristic / DQN |
| P | Pause |
| R | Restart |
| + / − | AI playback speed |
| Esc | Quit |

## How to train

Two presets:

| Preset | Meaning |
|--------|---------|
| `improved` (default) | Double DQN, LayerNorm, lr=`5e-4`, ε-decay 15k steps, **delta-based** rewards |
| `original` | Pre-audit ablation: vanilla DQN, lr=`1e-3`, absolute-height reward shaping |

```bash
# Recommended
python -m tetris_rl.training.train_dqn --preset improved --episodes 5000 --seed 42

# Ablation (legacy shaping + hparams)
python -m tetris_rl.training.train_dqn --preset original --episodes 400 --seed 42 \
  --checkpoint-dir models/original --log-dir results/train_original
```

Useful flags: `--device cpu|mps|cuda`, `--resume PATH`, `--double-dqn` / `--no-double-dqn`, `--no-tb`.

- Checkpoints: `models/*/dqn_best.pt`, `dqn_latest.pt`, `dqn_final.pt`
- Equal-budget A/B script: `bash scripts/run_ab_experiment.sh 400`

## How to evaluate all agents

```bash
# Single agent (greedy / no exploration)
python -m tetris_rl.evaluation.evaluate --agent random --games 100 --seed 0
python -m tetris_rl.evaluation.evaluate --agent heuristic --games 100 --seed 0
python -m tetris_rl.evaluation.evaluate --agent dqn --model models/improved/dqn_best.pt --games 100 --seed 0

# Full A/B: random + heuristic + original DQN + improved DQN
python -m tetris_rl.evaluation.compare_ab --games 100 --seed 0 \
  --original-model models/original/dqn_best.pt \
  --improved-model models/improved/dqn_best.pt \
  --out-dir results/ab_comparison
```

Outputs: summary CSV/JSON, per-game CSVs, charts (`mean_scores.png`, `score_boxplot.png`, `mean_lines.png`).

Eval metrics use **game score / lines cleared** (not shaped training reward). Policies run with ε=0.

## How to launch the AI demo

```bash
python -m tetris_rl.ui.pygame_app --mode human
python -m tetris_rl.ui.pygame_app --mode heuristic
python -m tetris_rl.ui.pygame_app --mode dqn --model models/improved/dqn_best.pt
```

## RL design

### DQN (algorithms implemented)

Placement-action value learning (custom PyTorch; not SB3 DQN):

1. Enumerate legal hard-drop placements.
2. Compute state-action features φ(s, a) for each (resulting board + piece info).
3. Q-network MLP: Q(s, a) = f_θ(φ(s, a)) (optional input LayerNorm in `improved`).
4. ε-greedy selection during training; greedy at eval.
5. Experience replay + Huber (Smooth L1) loss + gradient clipping + hard target sync.
6. Bootstrap: max over legal next actions; **Double DQN** on by default in `improved`
   (online argmax, target evaluation).
7. Terminal → bootstrap target 0.

Audit notes (`results/AUDIT.md`): Bellman/next-action handling were already correct; main learning bugs were reward shaping (absolute height tax) and terminal survival bonus on no-move game-overs.

### State representation

Observation vector includes:

- Board occupancy (10×20)
- Current piece one-hot
- Next (preview) piece one-hot
- Normalized column heights
- Aggregate height, holes, bumpiness, max height
- Lines cleared on the previous step (normalized)

The agent never sees future bag pieces beyond the single preview.

### Action space

Variable legal placements each turn, encoded as indices `0 .. n_legal-1` with a boolean mask of length `MAX_ACTIONS` (40). Agents must only select masked-legal indices. The DQN never scores illegal slots; it only evaluates enumerated placements.

### Reward function (training)

**Improved (default):** delta-based board terms — changes in holes / aggregate height / bumpiness, plus line-clear bonuses, survival bonus, and a large game-over penalty. Filling holes and lowering the stack are rewarded.

**Original (ablation):** absolute aggregate-height penalty every step (dominates returns; kept for comparison only).

**Training reward ≠ evaluation metrics.** Published tables use engine **score** and **lines cleared**.

## Testing

```bash
pytest -q
```

Coverage includes engine, env, rewards (including audit fixes), DQN forward/Double-DQN update, and checkpoint save/load.

## Measured results (executed locally)

### A/B comparison — 100 shared seeded games

Protocol: `--seed 0`, `--max-pieces 500`, ε=0 for DQN agents.

| Agent | Mean score | Median | Best | Mean lines | Mean pieces | Std score |
|-------|------------|--------|------|------------|-------------|-----------|
| Random | 8.0 | 0 | 100 | 0.08 | 25.7 | 27.3 |
| Heuristic | 21153.0 | 21300 | 22800 | 194.0 | 496.4 | 1413.8 |
| DQN original | 566.0 | 500 | 2300 | 5.46 | 48.5 | 405.1 |
| DQN improved | **939.0** | **900** | **2400** | **9.08** | **59.9** | 500.1 |

**Training budgets (verified):**
- **Original:** `--preset original`, 400 episodes, seed 42, max 200 pieces/ep → `models/original/dqn_best.pt`
- **Improved:** `--preset improved`, continued training to ε=0.05 (~19k env steps) → `models/improved/dqn_best.pt`

**Findings (measured, not claimed beyond this):**
- Improved DQN beats original DQN and random on this protocol.
- Neither DQN approaches the heuristic within this budget; heuristic rarely tops out before the 500-piece cap.
- Artifacts: `results/ab_comparison/` (CSV/JSON + charts). Code audit: `results/AUDIT.md`.

### Reproduce

```bash
bash scripts/run_ab_experiment.sh 400   # train both presets + compare
# or evaluate existing checkpoints:
python -m tetris_rl.evaluation.compare_ab --games 100 --seed 0 \
  --original-model models/original/dqn_best.pt \
  --improved-model models/improved/dqn_best.pt
```

## Limitations and next steps

- No HOLD / full SRS / T-spins / combo scoring
- Placement actions only
- Heuristic still much stronger; try longer `--episodes 5000`, prioritized replay, or PPO
- Resume training does not restore the replay buffer
- Eval `--max-pieces` cap inflates heuristic “survival” vs natural top-out

