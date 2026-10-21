#!/usr/bin/env python3
"""Phase-1 setup verification for the extended reward experiment (read-only)."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tetris_rl.agents.dqn_agent import DQNAgent  # noqa: E402
from tetris_rl.environment import TetrisEnv  # noqa: E402
from tetris_rl.features import board_metrics  # noqa: E402
from tetris_rl.rewards import (  # noqa: E402
    HeuristicAlignedRewardConfig,
    RewardConfig,
    shaped_reward,
)
from tetris_rl.training import diagnostics as dg  # noqa: E402
from tetris_rl.training.train_dqn import (  # noqa: E402
    VALIDATION_SEEDS,
    build_config,
    parse_args,
)

OUT = ROOT / "results" / "exp_reward_extended"
MAX_STEPS = 20_000
EXP_MODEL_DIR = ROOT / "models" / "exp_reward_extended"
EXP_RESULT_DIR = ROOT / "results" / "exp_reward_extended"
FROZEN = ROOT / "models" / "frozen" / "dqn_best_val1320.pt"
IMPROVED = ROOT / "models" / "improved" / "dqn_best.pt"
FROZEN_SHA = "a7351f49f59d0f1808c163a95877c5db8148dab904c10e731fd8e4a867303b1e"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    checks: dict = {}

    # 2. identical architecture / optimisation / action space / budget -----------
    arms = {}
    for mode in ("delta", "heuristic_aligned"):
        a = parse_args(["--preset", "improved", "--reward-mode", mode, "--seed", "42",
                        "--max-steps", str(MAX_STEPS), "--device", "cpu"])
        cfg, rcfg = build_config(a, "cpu")
        agent = DQNAgent(cfg)
        env = TetrisEnv(seed=0, reward_config=rcfg)
        arms[mode] = {
            "dqn_config": asdict(cfg),
            "reward_class": type(rcfg).__name__,
            "reward_config": rcfg.__dict__,
            "param_shapes": [tuple(p.shape) for p in agent.online.parameters()],
            "n_params": int(sum(p.numel() for p in agent.online.parameters())),
            "action_space": str(env.action_space),
            "obs_space": str(env.observation_space.shape),
            "max_steps": a.max_steps, "max_pieces": a.max_pieces, "train_freq": a.train_freq,
            "eval_games": a.eval_games, "gamma": a.gamma,
        }
    d, h = arms["delta"], arms["heuristic_aligned"]
    same_keys = ["dqn_config", "param_shapes", "n_params", "action_space", "obs_space",
                 "max_steps", "max_pieces", "train_freq", "eval_games", "gamma"]
    checks["identical_non_reward_settings"] = {k: d[k] == h[k] for k in same_keys}
    checks["reward_configs_differ_only_in"] = sorted(
        k for k in d["reward_config"] if d["reward_config"][k] != h["reward_config"][k]
    )
    checks["arm_settings"] = arms

    # 3. reward mode applied to transitions ---------------------------------------
    rows = []
    envs = {m: TetrisEnv(seed=100_000, reward_config=c)
            for m, c in (("delta", RewardConfig()), ("heuristic_aligned", HeuristicAlignedRewardConfig()))}
    for e in envs.values():
        e.reset(seed=100_000)
    ok = True
    rng = np.random.default_rng(0)
    for _ in range(25):
        if envs["delta"].engine.game_over:
            break
        a = int(rng.integers(0, len(envs["delta"].legal_placements())))
        row = {}
        for m, e in envs.items():
            before = e.engine.copy_board()
            _, r, term, _, info = e.step(a)
            expected = shaped_reward(e.engine, before, e.engine.copy_board(),
                                     info["lines_cleared"], e.engine.game_over, config=e.reward_config)
            ok &= abs(r - expected) < 1e-9
            row[m] = float(r)
        rows.append(row)
    checks["reward_wiring"] = {
        "env_reward_equals_shaped_reward_for_own_config": bool(ok),
        "same_actions_but_different_rewards": bool(
            any(abs(r["delta"] - r["heuristic_aligned"]) > 1e-6 for r in rows)),
        "sample_rewards_first5": rows[:5],
        "train_loop_passes_reward_config": "reward_config=reward_cfg" in
            (ROOT / "tetris_rl/training/train_dqn.py").read_text(),
    }

    # 4/5. epsilon is env-step based; steps needed ---------------------------------
    cfg = build_config(parse_args(["--preset", "improved"]), "cpu")[0]
    obs_src = (ROOT / "tetris_rl/agents/dqn_agent.py").read_text()
    eps_by_steps = "frac = min(1.0, self.total_steps /" in obs_src and "_decay_epsilon()" in obs_src
    per_run = []
    for mode in ("delta", "heuristic_aligned"):
        for seed in (42, 43):
            p = ROOT / "results" / "exp_reward_ep400" / mode / f"seed{seed}" / "training_metrics.csv"
            rws = list(csv.DictReader(p.open()))
            # seed-42 logs hold a duplicate appended rerun; keep the first 400-episode block
            rws = rws[:400]
            steps = np.array([int(r["steps"]) for r in rws])
            eps = np.array([float(r["epsilon"]) for r in rws])
            pred = np.maximum(cfg.epsilon_end, 1.0 - 0.95 * np.minimum(1.0, steps / cfg.epsilon_decay_steps))
            last_steps = int(steps[-1])
            spe = last_steps / len(rws)
            per_run.append({
                "mode": mode, "seed": seed, "episodes": len(rws), "steps_at_ep400": last_steps,
                "eps_at_ep400": float(eps[-1]),
                "max_abs_eps_err_vs_step_formula": float(np.max(np.abs(eps - pred))),
                "mean_steps_per_episode": spe,
                "extra_steps_to_eps_0.05": max(0, cfg.epsilon_decay_steps - last_steps),
                "extra_episodes_to_eps_0.05_at_current_len": max(0, cfg.epsilon_decay_steps - last_steps) / spe,
            })
    checks["epsilon"] = {
        "decay_steps": cfg.epsilon_decay_steps, "eps_end": cfg.epsilon_end,
        "decay_is_function_of_total_env_steps": bool(eps_by_steps),
        "linear_formula": "eps = 1 - 0.95 * min(1, steps/15000)",
        "eps_at_20000_steps": 0.05,
        "steps_at_floor_in_20k_budget": MAX_STEPS - cfg.epsilon_decay_steps,
        "ep400_runs": per_run,
    }

    # 6. seed disjointness ---------------------------------------------------------
    max_train_seed = 43 + 3000  # seed (<=43) + episode index; 20k steps << 3000 episodes at >=7 steps
    train_seeds = set(range(43, max_train_seed))
    val = set(VALIDATION_SEEDS)
    diag = set(dg.DIAG_HEURISTIC_SEEDS) | set(dg.DIAG_RANDOM_SEEDS)
    test = set(range(0, 100))
    checks["seeds"] = {
        "train_env_seeds": "seed+episode (seed in {42,43}); episode count <~2000 for 20k steps",
        "val_range": [min(val), max(val)],
        "diag_ranges": [[min(dg.DIAG_HEURISTIC_SEEDS), max(dg.DIAG_HEURISTIC_SEEDS)],
                        [min(dg.DIAG_RANDOM_SEEDS), max(dg.DIAG_RANDOM_SEEDS)]],
        "train_val_disjoint": not (train_seeds & val),
        "train_diag_disjoint": not (train_seeds & diag),
        "val_test_disjoint": not (val & test),
        "val_diag_disjoint": not (val & diag),
        "NOTE_train_overlaps_test_seeds_0_99": sorted(train_seeds & test)[:3] + ["..."],
        "NOTE": "Training episode seeds 43..99 coincide with test seeds 43..99 "
                "(same piece sequences, random-play era). Test seeds are NOT used in this experiment.",
        "NOTE_seed42_vs_seed43": "env seed streams are offset by one episode (seed+ep), so the "
                                 "two seeds share almost identical game sequences; network init / "
                                 "exploration RNG differ.",
    }

    # 7. protected checkpoints -----------------------------------------------------
    checks["protected_checkpoints"] = {
        "frozen_sha256": sha256(FROZEN), "improved_best_sha256": sha256(IMPROVED),
        "frozen_matches_recorded": sha256(FROZEN) == FROZEN_SHA,
        "improved_best_same_as_frozen": sha256(IMPROVED) == FROZEN_SHA,
    }

    # 8. no overwrite --------------------------------------------------------------
    checks["output_paths"] = {
        "models_dir": str(EXP_MODEL_DIR.relative_to(ROOT)),
        "results_dir": str(EXP_RESULT_DIR.relative_to(ROOT)),
        "models_dir_exists": EXP_MODEL_DIR.exists(),
        "results_dir_has_training_dirs": any((EXP_RESULT_DIR / m).exists()
                                             for m in ("delta", "heuristic_aligned")),
    }

    # resume fairness ----------------------------------------------------------------
    p42 = ROOT / "results/exp_reward_ep400/delta/seed42/training_metrics.csv"
    checks["resume_decision"] = {
        "seed42_ep400_logs_rows": sum(1 for _ in p42.open()) - 1,
        "reason": "ep400 seed-42 logs contain a duplicated appended rerun, and train() "
                  "restarts episode numbering (env seed = seed+ep) on resume, replaying the same "
                  "early games. Fresh initialisation is cleaner and costs ~10 min/run.",
        "decision": "fresh init for all four runs",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1_verification.json").write_text(json.dumps(checks, indent=2, default=str))
    print(json.dumps(checks, indent=2, default=str))


if __name__ == "__main__":
    main()
