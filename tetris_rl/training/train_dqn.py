"""
CLI entry point for training the placement-action DQN.

Example:
    python -m tetris_rl.training.train_dqn --episodes 5000 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from tetris_rl.agents.dqn_agent import (
    DQNAgent,
    DQNConfig,
    improved_dqn_config,
    original_dqn_config,
)
from tetris_rl.environment import TetrisEnv
from tetris_rl.features import placement_feature_batch
from tetris_rl.rewards import RewardConfig, reward_config_from_name
from tetris_rl.training.demos import DemoSet, bc_pretrain, heuristic_choice
from tetris_rl.training.imitation import ImitationRegularizer, demo_metrics, fixed_subset
from tetris_rl.training.diagnostics import (
    build_diagnostic_states,
    evaluate_greedy_extended,
    probe_diagnostic_states,
)

# Fixed validation seeds for model selection (disjoint from final test seeds 0..N).
VALIDATION_SEEDS: tuple[int, ...] = tuple(range(100_000, 100_050))
# Final held-out test seed base used by evaluation.compare / compare_ab.
TEST_SEED_BASE = 0

# Per-gradient-step statistics recorded by DQNAgent._add_imitation (window means).
IMIT_WINDOW_KEYS: tuple[str, ...] = (
    "td_loss", "imit_loss", "imit_weighted", "total_loss",
    "imit_top1", "imit_margin", "imit_expert_prob",
    "gn_td", "gn_imit_unweighted", "gn_imit_weighted", "grad_cos_td_imit",
)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Tetris placement-action DQN")
    p.add_argument("--episodes", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | mps (auto if omitted)")
    p.add_argument(
        "--preset",
        type=str,
        default="improved",
        choices=["original", "improved"],
        help="original = pre-audit hparams+rewards; improved = post-audit defaults",
    )
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--min-buffer-size", type=int, default=None)
    p.add_argument("--target-update-freq", type=int, default=None)
    p.add_argument("--epsilon-start", type=float, default=None)
    p.add_argument("--epsilon-end", type=float, default=None)
    p.add_argument("--epsilon-decay-steps", type=int, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--double-dqn", action="store_true")
    p.add_argument("--no-double-dqn", action="store_true")
    p.add_argument("--train-freq", type=int, default=1, help="Gradient steps per env step")
    p.add_argument("--eval-every", type=int, default=100, help="Episodes between greedy evals")
    p.add_argument(
        "--eval-games",
        type=int,
        default=50,
        help="Validation games (capped to len(VALIDATION_SEEDS)=50)",
    )
    p.add_argument("--max-pieces", type=int, default=500, help="Safety cap per episode")
    p.add_argument("--checkpoint-dir", type=str, default="models")
    p.add_argument("--log-dir", type=str, default="results/train_logs")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--tb", action="store_true", help="Enable TensorBoard logging")
    p.add_argument("--no-tb", action="store_true", help="Disable TensorBoard (default on)")
    p.add_argument(
        "--reward-mode",
        type=str,
        default=None,
        choices=["delta", "legacy", "heuristic_aligned"],
        help="Override reward shaping. heuristic_aligned uses Dellacherie weights.",
    )
    # Step-budget mode (extended experiments). When --max-steps is set, training
    # stops after exactly that many environment steps (possibly mid-episode) and
    # evaluation / checkpointing are triggered by env-step counts, not episodes.
    p.add_argument("--max-steps", type=int, default=None, help="Stop after N env steps")
    p.add_argument(
        "--eval-every-steps", type=int, default=2500, help="Env steps between evals (step mode)"
    )
    p.add_argument(
        "--save-every-steps",
        type=int,
        default=None,
        help="Env steps between periodic weight checkpoints (default: eval-every-steps)",
    )
    # Heuristic-guided training (step mode). All default to "off" (= baseline behaviour).
    p.add_argument("--heuristic-mix-start", type=float, default=0.0,
                   help="P(take heuristic placement) at step 0; decays linearly to 0")
    p.add_argument("--heuristic-mix-steps", type=int, default=15_000,
                   help="Env steps over which the heuristic probability decays to 0")
    p.add_argument("--demo-path", type=str, default=None, help="Heuristic demo .npz (train)")
    p.add_argument("--demo-val-path", type=str, default=None, help="Heuristic demo .npz (held-out)")
    p.add_argument("--demo-prefill", action="store_true", help="Insert demos into the replay buffer")
    p.add_argument("--bc-pretrain-steps", type=int, default=0,
                   help="Behaviour-cloning (listwise CE + TD) steps on demos before RL")
    p.add_argument("--bc-td-weight", type=float, default=1.0)
    p.add_argument("--eval-at-start", action="store_true", help="Also evaluate at env step 0")
    # Imitation-regularised Double DQN (all default off = baseline behaviour).
    p.add_argument("--init-from", type=str, default=None,
                   help="Initialise online/target (+Adam) from this checkpoint; counters/eps/buffer are fresh")
    p.add_argument("--init-fresh-optimizer", action="store_true",
                   help="With --init-from: do not load the checkpoint's Adam state")
    p.add_argument("--imitation-weight", type=float, default=None,
                   help="lambda in L = L_TD + lambda * L_imit (needs --demo-path). 0 = log-only control")
    p.add_argument("--imitation-batch-size", type=int, default=64,
                   help="Demo states per imitation update")
    p.add_argument("--grad-diag-every", type=int, default=50,
                   help="Every N grad steps, record per-loss gradient norms (instrumentation only)")
    p.add_argument("--train-seed-base", type=int, default=None,
                   help="Training episode seeds = base + episode (default: --seed + episode)")
    p.add_argument("--eval-at-steps", type=str, default=None,
                   help="Comma-separated env steps for validation + checkpoint (overrides --eval-every-steps)")
    return p.parse_args(argv)


def build_config(args: argparse.Namespace, device: str) -> tuple[DQNConfig, RewardConfig]:
    """Build DQN + reward configs from --preset and CLI overrides."""
    factory = original_dqn_config if args.preset == "original" else improved_dqn_config
    overrides: dict = {"gamma": args.gamma, "device": device, "seed": args.seed}
    for key in (
        "lr",
        "batch_size",
        "buffer_size",
        "min_buffer_size",
        "target_update_freq",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_steps",
        "grad_clip",
    ):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val

    cfg = factory(**overrides)
    if args.double_dqn:
        cfg.double_dqn = True
    if args.no_double_dqn:
        cfg.double_dqn = False

    if args.reward_mode is not None:
        reward_cfg = reward_config_from_name(args.reward_mode)
    elif args.preset == "original":
        reward_cfg = reward_config_from_name("legacy")
    else:
        reward_cfg = reward_config_from_name("delta")
    return cfg, reward_cfg


def resolve_device(requested: Optional[str]) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def evaluate_greedy(
    agent: DQNAgent,
    max_pieces: int,
    seeds: Optional[Sequence[int]] = None,
    n_games: Optional[int] = None,
) -> dict:
    """
    Run greedy (ε=0) games on fixed validation seeds.

    Default seeds are VALIDATION_SEEDS (disjoint from final test seeds starting
    at TEST_SEED_BASE). Does not mutate exploration schedule permanently.
    """
    if seeds is None:
        seeds = VALIDATION_SEEDS
    if n_games is not None:
        seeds = list(seeds)[:n_games]

    old_eps = agent.epsilon
    agent.epsilon = 0.0
    scores, lines, pieces, topped = [], [], [], []
    for seed in seeds:
        env = TetrisEnv(seed=int(seed))
        _, info = env.reset(seed=int(seed))
        terminated = truncated = False
        while not terminated and not truncated:
            if info["n_legal"] == 0:
                break
            if env.engine.pieces_placed >= max_pieces:
                truncated = True
                break
            action = agent.select_action(env, explore=False)
            _, _, terminated, trunc_env, info = env.step(action)
            truncated = bool(trunc_env) or env.engine.pieces_placed >= max_pieces
        scores.append(env.engine.score)
        lines.append(env.engine.lines_cleared_total)
        pieces.append(env.engine.pieces_placed)
        topped.append(bool(env.engine.game_over) and not truncated)
    agent.epsilon = old_eps
    scores_a = np.asarray(scores, dtype=np.float64)
    lines_a = np.asarray(lines, dtype=np.float64)
    pieces_a = np.asarray(pieces, dtype=np.float64)
    return {
        "mean_score": float(np.mean(scores_a)),
        "median_score": float(np.median(scores_a)),
        "std_score": float(np.std(scores_a, ddof=1) if len(scores_a) > 1 else 0.0),
        "mean_lines": float(np.mean(lines_a)),
        "median_lines": float(np.median(lines_a)),
        "mean_pieces": float(np.mean(pieces_a)),
        "top_out_rate": float(np.mean(topped)),
        "n_games": len(scores_a),
    }


def probe_fixed_state_q(agent: DQNAgent, seeds: Optional[Sequence[int]] = None) -> dict:
    """Mean greedy Q on the first state of a few fixed validation seeds."""
    if seeds is None:
        seeds = VALIDATION_SEEDS[:8]
    q_max, q_mean = [], []
    for seed in seeds:
        env = TetrisEnv(seed=int(seed))
        env.reset(seed=int(seed))
        feats = placement_feature_batch(env.engine, env.legal_placements())
        q = agent.q_values(feats, agent.online)
        if q.size == 0:
            continue
        q_max.append(float(np.max(q)))
        q_mean.append(float(np.mean(q)))
    return {
        "q_max_mean": float(np.mean(q_max)) if q_max else float("nan"),
        "q_mean_mean": float(np.mean(q_mean)) if q_mean else float("nan"),
    }


def train(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg, reward_cfg = build_config(args, device)
    print(
        f"preset={args.preset} reward={type(reward_cfg).__name__} device={device} "
        f"lr={cfg.lr} double_dqn={cfg.double_dqn} "
        f"eps_decay={cfg.epsilon_decay_steps} layer_norm={cfg.use_layer_norm}"
    )

    if args.resume:
        agent = DQNAgent.from_checkpoint(args.resume, device=device, explore=True)
        print(
            f"Resumed from {args.resume} (steps={agent.total_steps}, eps={agent.epsilon:.3f}, "
            f"best_eval={agent.best_eval_score:.1f}, buffer={len(agent.buffer)})"
        )
        # Checkpoint config wins unless CLI explicitly overrides --lr.
        if args.lr is not None:
            agent.config.lr = args.lr
            for group in agent.optimizer.param_groups:
                group["lr"] = args.lr
            print(f"Overrode optimizer lr to {args.lr}")
    else:
        agent = DQNAgent(cfg)

    if args.init_from:
        if args.resume:
            raise SystemExit("--init-from and --resume are mutually exclusive")
        agent.load_init_weights(args.init_from, load_optimizer=not args.init_fresh_optimizer)
        print(f"Initialised weights from {args.init_from} (fresh counters, eps={agent.epsilon:.3f})")

    use_tb = args.tb or not args.no_tb
    writer = None
    if use_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(log_dir / "tb"))
        except Exception as exc:  # pragma: no cover
            print(f"TensorBoard unavailable ({exc}); continuing without it.")

    step_mode = args.max_steps is not None
    csv_path = log_dir / "training_metrics.csv"
    write_header = not csv_path.exists()
    csv_file = csv_path.open("a", newline="")
    train_fields = [
        "episode",
        "score",
        "lines",
        "pieces",
        "reward",
        "epsilon",
        "loss",
        "q_mean",
        "td_error",
        "steps",
        "terminated",
        "truncated",
    ]
    if step_mode:
        train_fields += ["grad_norm", "budget_cut"]
    csv_writer = csv.DictWriter(csv_file, fieldnames=train_fields)
    if write_header:
        csv_writer.writeheader()

    val_csv_path = log_dir / "val_metrics.csv"
    val_write_header = not val_csv_path.exists()
    val_csv_file = val_csv_path.open("a", newline="")
    val_csv_writer = csv.DictWriter(
        val_csv_file,
        fieldnames=[
            "episode",
            "mean_score",
            "median_score",
            "mean_lines",
            "mean_pieces",
            "top_out_rate",
            "q_max_mean",
            "loss",
            "td_error",
        ],
    )
    if val_write_header:
        val_csv_writer.writeheader()

    meta = {
        "preset": args.preset,
        "reward_mode": args.reward_mode,
        "reward_class": type(reward_cfg).__name__,
        "config": cfg.__dict__,
        "reward_config": reward_cfg.__dict__,
        "episodes": args.episodes,
        "seed": args.seed,
        "device": device,
        "validation_seeds": list(VALIDATION_SEEDS),
        "test_seed_base": TEST_SEED_BASE,
        "max_steps": args.max_steps,
        "eval_every_steps": args.eval_every_steps if step_mode else None,
        "save_every_steps": (args.save_every_steps or args.eval_every_steps) if step_mode else None,
        "max_pieces": args.max_pieces,
        "train_freq": args.train_freq,
        "heuristic_mix_start": args.heuristic_mix_start,
        "heuristic_mix_steps": args.heuristic_mix_steps,
        "demo_path": args.demo_path,
        "demo_prefill": args.demo_prefill,
        "bc_pretrain_steps": args.bc_pretrain_steps,
        "bc_td_weight": args.bc_td_weight,
        "init_from": args.init_from,
        "init_fresh_optimizer": args.init_fresh_optimizer,
        "imitation_weight": args.imitation_weight,
        "imitation_batch_size": args.imitation_batch_size,
        "train_seed_base": args.train_seed_base,
        "eval_at_steps": args.eval_at_steps,
    }
    (log_dir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str))

    # Preserve best score across resumed runs.
    best_eval = agent.best_eval_score
    t0 = time.time()
    n_val = min(args.eval_games, len(VALIDATION_SEEDS))

    # --- step-budget mode state -------------------------------------------
    eval_every_steps = args.eval_every_steps
    save_every_steps = args.save_every_steps or args.eval_every_steps
    next_eval_step = (agent.total_steps // eval_every_steps + 1) * eval_every_steps
    next_save_step = (agent.total_steps // save_every_steps + 1) * save_every_steps
    # Explicit validation schedule (also used for checkpoints, so every validated
    # state of the policy is preserved).
    sched: list[int] = (
        sorted({int(s) for s in args.eval_at_steps.split(",") if s.strip()})
        if args.eval_at_steps
        else []
    )
    sched = [s for s in sched if s > agent.total_steps]
    if sched:
        next_eval_step = next_save_step = sched[0]
    eval_csv_path = log_dir / "eval_metrics.csv"
    eval_csv_file = None
    eval_csv_writer = None
    diag_states = None
    win: dict[str, list[float]] = {"loss": [], "td": [], "gn": [], "q": []}
    for k in IMIT_WINDOW_KEYS:
        win[k] = []
    eval_seconds_total = 0.0
    budget_cut = False
    n_clipped = 0
    n_grad_steps_window = 0
    if step_mode:
        diag_states = build_diagnostic_states()
        print(
            f"step mode: max_steps={args.max_steps} eval_every={eval_every_steps} "
            f"diag_states heuristic={len(diag_states.heuristic)} random={len(diag_states.random)}"
        )

    def step_eval() -> dict:
        """Greedy eval + diagnostics at the current env-step; writes CSVs, ckpts."""
        nonlocal eval_csv_file, eval_csv_writer, best_eval, eval_seconds_total
        nonlocal n_clipped, n_grad_steps_window
        te = time.time()
        metrics = evaluate_greedy_extended(
            agent, args.max_pieces, reward_cfg, seeds=VALIDATION_SEEDS[:n_val], gamma=cfg.gamma
        )
        probe = probe_diagnostic_states(agent, diag_states)
        eval_dt = time.time() - te
        eval_seconds_total += eval_dt
        elapsed = time.time() - t0

        def _m(x):
            return float(np.mean(x)) if x else float("nan")

        row = {
            "env_steps": agent.total_steps,
            "episodes": agent.episodes_trained,
            "epsilon": agent.epsilon,
            "elapsed_s": elapsed,
            "eval_s": eval_dt,
            **metrics,
            **probe,
            "mix_p": mix_p_now(),
            "h_action_frac": (mix_cnt["h"] / mix_cnt["n"]) if mix_cnt["n"] else float("nan"),
            "train_loss": _m(win["loss"]),
            "train_td_error": _m(win["td"]),
            "train_q_mean": _m(win["q"]),
            "grad_norm_mean": _m(win["gn"]),
            "grad_norm_max": float(np.max(win["gn"])) if win["gn"] else float("nan"),
            "grad_clip_frac": (n_clipped / n_grad_steps_window) if n_grad_steps_window else float("nan"),
            "n_grad_steps_window": n_grad_steps_window,
        }
        # Imitation-regularisation diagnostics (training-window means) ...
        for k in IMIT_WINDOW_KEYS:
            row[f"w_{k}"] = _m(win[k])
        # ... and expert-preference retention measured on FIXED demo states (no grad).
        for name, (dset, idx) in demo_probe_sets.items():
            for mk, mv in demo_metrics(agent.online, dset, idx).items():
                row[f"{name}_{mk}"] = mv
        if eval_csv_writer is None:
            eval_csv_file = eval_csv_path.open("a", newline="")
            eval_csv_writer = csv.DictWriter(eval_csv_file, fieldnames=list(row.keys()))
            if eval_csv_path.stat().st_size == 0:
                eval_csv_writer.writeheader()
        eval_csv_writer.writerow(row)
        eval_csv_file.flush()
        val_csv_writer.writerow(
            {
                "episode": agent.episodes_trained,
                "mean_score": metrics["mean_score"],
                "median_score": metrics["median_score"],
                "mean_lines": metrics["mean_lines"],
                "mean_pieces": metrics["mean_pieces"],
                "top_out_rate": metrics["top_out_rate"],
                "q_max_mean": probe["hs_q_max_mean"],
                "loss": row["train_loss"],
                "td_error": row["train_td_error"],
            }
        )
        val_csv_file.flush()
        print(
            f"  [val@{agent.total_steps} steps, ep={agent.episodes_trained}, eps={agent.epsilon:.3f}] "
            f"mean={metrics['mean_score']:.1f} med={metrics['median_score']:.0f} "
            f"lines={metrics['mean_lines']:.2f} pieces={metrics['mean_pieces']:.1f} "
            f"top_out={metrics['top_out_rate']:.2f} holes/mv={metrics['holes_created_per_move']:.2f} "
            f"agree_hs={probe['hs_top1_agreement']:.2f} rho_hs={probe['hs_spearman']:.2f} "
            f"Qmax_hs={probe['hs_q_max_mean']:.2f} loss={row['train_loss']:.3f} "
            f"td={row['w_td_loss']:.3f} imit={row['w_imit_loss']:.3f} "
            f"demo_val_top1={row.get('demo_val_top1', float('nan')):.2f} "
            f"gn={row['grad_norm_mean']:.2f} (eval {eval_dt:.1f}s, elapsed {elapsed:.0f}s)"
        )
        for v in win.values():
            v.clear()
        mix_cnt["h"] = mix_cnt["n"] = 0
        n_clipped = 0
        n_grad_steps_window = 0
        if metrics["mean_score"] >= best_eval:
            best_eval = metrics["mean_score"]
            agent.best_eval_score = best_eval
            agent.save(checkpoint_dir / "dqn_best.pt", include_buffer=False)
            row["_new_best"] = True
        return row

    # --- heuristic guidance setup (off by default) --------------------------
    mix_cnt = {"h": 0, "n": 0}
    demo_probe_sets: dict = {}  # name -> (DemoSet, fixed index array)
    mix_rng = np.random.default_rng(args.seed + 777)  # separate from agent.rng
    tie_rng = np.random.default_rng(args.seed + 778)

    def mix_p_now() -> float:
        if args.heuristic_mix_start <= 0:
            return 0.0
        return args.heuristic_mix_start * max(0.0, 1.0 - agent.total_steps / max(1, args.heuristic_mix_steps))

    demo_info: dict = {}
    if args.demo_path and (
        args.demo_prefill or args.bc_pretrain_steps > 0 or args.imitation_weight is not None
    ):
        demos = DemoSet.load(args.demo_path)
        val_demos = DemoSet.load(args.demo_val_path) if args.demo_val_path else None
        demo_probe_sets["demo_train"] = (demos, fixed_subset(demos, 2000, seed=11))
        if val_demos is not None:
            demo_probe_sets["demo_val"] = (val_demos, fixed_subset(val_demos, 2000, seed=12))
        if args.imitation_weight is not None:
            agent.attach_imitation(
                ImitationRegularizer(
                    demos, args.imitation_weight, batch_size=args.imitation_batch_size,
                    seed=args.seed + 4242, device=device,
                ),
                grad_diag_every=args.grad_diag_every,
            )
            demo_info["imitation"] = {
                "weight": args.imitation_weight,
                "batch_size": args.imitation_batch_size,
                "n_demo_transitions": len(demos),
                "n_demo_games": demos.mean_return_stats()["n_games"],
            }
            print(f"imitation regulariser: lambda={args.imitation_weight} on {len(demos)} demo states")
        demo_info["stats"] = demos.mean_return_stats()
        print("demos:", demo_info["stats"])
        if args.bc_pretrain_steps > 0:
            tb = time.time()
            bc_log = bc_pretrain(
                agent, demos, args.bc_pretrain_steps, td_weight=args.bc_td_weight,
                val_demos=val_demos, seed=args.seed + 999,
            )
            demo_info["bc_log"] = bc_log
            demo_info["bc_seconds"] = time.time() - tb
            for rec in bc_log:
                print("  [bc]", {k: round(v, 4) for k, v in rec.items()})
        if args.demo_prefill:
            demo_info["prefilled"] = demos.fill_buffer(agent.buffer)
            print(f"prefilled replay buffer with {demo_info['prefilled']} demo transitions")
        (log_dir / "demo_info.json").write_text(json.dumps(demo_info, indent=2, default=str))
        t0 = time.time()  # exclude demo loading/pretraining from run-time; reported in demo_info

    if step_mode and args.eval_at_start:
        step_eval()

    max_episodes = 10**9 if step_mode else args.episodes
    for ep in range(1, max_episodes + 1):
        ep_seed = (args.train_seed_base if args.train_seed_base is not None else args.seed) + ep
        env = TetrisEnv(seed=ep_seed, reward_config=reward_cfg)
        _, info = env.reset(seed=ep_seed)
        ep_reward = 0.0
        losses: list[float] = []
        q_means: list[float] = []
        td_errors: list[float] = []
        grad_norms_ep: list[float] = []
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if info["n_legal"] == 0:
                break
            use_heuristic = args.heuristic_mix_start > 0 and mix_rng.random() < mix_p_now()
            if use_heuristic:
                feats_all = placement_feature_batch(env.engine, env.legal_placements())
                action = heuristic_choice(feats_all, tie_rng)
            else:
                action = agent.select_action(env, explore=True)
            mix_cnt["n"] += 1
            mix_cnt["h"] += int(use_heuristic)
            sa_feats = agent.features_for_action(env, action)
            _, reward, terminated, trunc_env, info = env.step(action)
            # Distinguish true terminal (game over) from piece-cap truncation.
            truncated = bool(trunc_env) or (env.engine.pieces_placed >= args.max_pieces)
            agent.observe(
                sa_feats,
                float(reward),
                env,
                terminated=bool(terminated),
                truncated=truncated,
            )
            ep_reward += float(reward)

            for _ in range(args.train_freq):
                if len(agent.buffer) >= cfg.min_buffer_size:
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)
                        stats = agent.last_train_stats
                        if "q_mean" in stats:
                            q_means.append(stats["q_mean"])
                        if "td_error" in stats:
                            td_errors.append(stats["td_error"])
                        if step_mode:
                            win["loss"].append(loss)
                            win["q"].append(stats["q_mean"])
                            win["td"].append(stats["td_error"])
                            for k in IMIT_WINDOW_KEYS:
                                if k in stats:
                                    win[k].append(stats[k])
                            gn = stats.get("grad_norm")
                            if gn is not None:
                                win["gn"].append(gn)
                                grad_norms_ep.append(gn)
                                n_grad_steps_window += 1
                                if gn > cfg.grad_clip:
                                    n_clipped += 1

            if step_mode:
                if agent.total_steps >= next_eval_step:
                    step_eval()
                    if sched:
                        sched.pop(0)
                        next_eval_step = sched[0] if sched else 10**12
                    else:
                        next_eval_step += eval_every_steps
                if agent.total_steps >= next_save_step:
                    agent.best_eval_score = best_eval
                    agent.save(
                        checkpoint_dir / f"dqn_step{agent.total_steps}.pt", include_buffer=False
                    )
                    if args.eval_at_steps:
                        # next_eval_step was already advanced above (same schedule)
                        next_save_step = next_eval_step
                    else:
                        next_save_step += save_every_steps
                if agent.total_steps >= args.max_steps:
                    budget_cut = not (terminated or truncated)
                    break

        agent.episodes_trained += 1
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_q = float(np.mean(q_means)) if q_means else float("nan")
        mean_td = float(np.mean(td_errors)) if td_errors else float("nan")
        row = {
            "episode": ep,
            "score": env.engine.score,
            "lines": env.engine.lines_cleared_total,
            "pieces": env.engine.pieces_placed,
            "reward": ep_reward,
            "epsilon": agent.epsilon,
            "loss": mean_loss,
            "q_mean": mean_q,
            "td_error": mean_td,
            "steps": agent.total_steps,
            "terminated": int(terminated),
            "truncated": int(truncated),
        }
        if step_mode:
            row["grad_norm"] = float(np.mean(grad_norms_ep)) if grad_norms_ep else float("nan")
            row["budget_cut"] = int(budget_cut)
        csv_writer.writerow(row)
        csv_file.flush()

        if writer is not None:
            writer.add_scalar("train/score", row["score"], agent.episodes_trained)
            writer.add_scalar("train/lines", row["lines"], agent.episodes_trained)
            writer.add_scalar("train/episode_reward", ep_reward, agent.episodes_trained)
            writer.add_scalar("train/epsilon", agent.epsilon, agent.episodes_trained)
            if losses:
                writer.add_scalar("train/loss", mean_loss, agent.episodes_trained)
            if q_means:
                writer.add_scalar("train/q_mean", mean_q, agent.episodes_trained)
            if td_errors:
                writer.add_scalar("train/td_error", mean_td, agent.episodes_trained)

        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - t0
            print(
                f"ep={ep:5d} score={row['score']:5d} lines={row['lines']:3d} "
                f"pieces={row['pieces']:3d} R={ep_reward:7.2f} eps={agent.epsilon:.3f} "
                f"loss={mean_loss:.4f} steps={agent.total_steps} "
                f"T/Tr={int(terminated)}/{int(truncated)} ({elapsed:.1f}s)"
            )

        if step_mode:
            if agent.total_steps >= args.max_steps:
                break
            continue

        if ep % args.save_every == 0:
            agent.best_eval_score = best_eval
            agent.save(checkpoint_dir / f"dqn_ep{ep}.pt")
            agent.save(checkpoint_dir / "dqn_latest.pt")

        if args.eval_every > 0 and ep % args.eval_every == 0:
            metrics = evaluate_greedy(agent, args.max_pieces, n_games=n_val)
            q_probe = probe_fixed_state_q(agent)
            print(
                f"  [val n={metrics['n_games']}] mean_score={metrics['mean_score']:.1f} "
                f"median={metrics['median_score']:.1f} "
                f"mean_lines={metrics['mean_lines']:.2f} "
                f"mean_pieces={metrics['mean_pieces']:.1f} "
                f"top_out={metrics['top_out_rate']:.2f} "
                f"q_max={q_probe['q_max_mean']:.2f} "
                f"(best={best_eval:.1f})"
            )
            val_csv_writer.writerow(
                {
                    "episode": ep,
                    "mean_score": metrics["mean_score"],
                    "median_score": metrics["median_score"],
                    "mean_lines": metrics["mean_lines"],
                    "mean_pieces": metrics["mean_pieces"],
                    "top_out_rate": metrics["top_out_rate"],
                    "q_max_mean": q_probe["q_max_mean"],
                    "loss": mean_loss,
                    "td_error": mean_td,
                }
            )
            val_csv_file.flush()
            if writer is not None:
                writer.add_scalar("val/mean_score", metrics["mean_score"], agent.episodes_trained)
                writer.add_scalar("val/mean_lines", metrics["mean_lines"], agent.episodes_trained)
                writer.add_scalar("val/q_max_mean", q_probe["q_max_mean"], agent.episodes_trained)
            if metrics["mean_score"] >= best_eval:
                best_eval = metrics["mean_score"]
                agent.best_eval_score = best_eval
                agent.save(checkpoint_dir / "dqn_best.pt")

    agent.best_eval_score = best_eval
    agent.save(checkpoint_dir / "dqn_final.pt")
    if not step_mode:
        agent.save(checkpoint_dir / "dqn_latest.pt")
    csv_file.close()
    val_csv_file.close()
    if eval_csv_file is not None:
        eval_csv_file.close()
    if step_mode:
        summary = {
            "reward_mode": args.reward_mode,
            "seed": args.seed,
            "device": device,
            "env_steps": agent.total_steps,
            "episodes_started": ep,
            "episodes_completed": ep - int(budget_cut),
            "final_epsilon": agent.epsilon,
            "grad_steps": agent.grad_steps,
            # Budgets are reported separately: env_steps counts RL interaction only.
            "rl_env_steps": agent.total_steps,
            "expert_demo_transitions_available": (
                len(demo_probe_sets["demo_train"][0]) if "demo_train" in demo_probe_sets else 0
            ),
            "expert_demo_env_steps_collected_offline": (
                len(demo_probe_sets["demo_train"][0]) if "demo_train" in demo_probe_sets else 0
            ),
            "imitation_weight": args.imitation_weight,
            "imitation_state_samples": (
                agent.grad_steps * args.imitation_batch_size if args.imitation_weight else 0
            ),
            "budget_cut_mid_episode": bool(budget_cut),
            "elapsed_s": time.time() - t0,
            "eval_s": eval_seconds_total,
            "train_s": time.time() - t0 - eval_seconds_total,
            "best_eval_score": best_eval,
        }
        (log_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
        print("run_summary:", json.dumps(summary))
    if writer is not None:
        writer.close()
    print(f"Training complete. Checkpoints in {checkpoint_dir} (best_eval={best_eval:.1f})")
    return checkpoint_dir / "dqn_final.pt"


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
