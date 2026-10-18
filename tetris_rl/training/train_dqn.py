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
from tetris_rl.rewards import LegacyRewardConfig, RewardConfig

# Fixed validation seeds for model selection (disjoint from final test seeds 0..N).
VALIDATION_SEEDS: tuple[int, ...] = tuple(range(100_000, 100_050))
# Final held-out test seed base used by evaluation.compare / compare_ab.
TEST_SEED_BASE = 0


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

    if args.preset == "original":
        reward_cfg: RewardConfig = LegacyRewardConfig()
    else:
        reward_cfg = RewardConfig()
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
    scores, lines, pieces = [], [], []
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
    agent.epsilon = old_eps
    return {
        "mean_score": float(np.mean(scores)),
        "mean_lines": float(np.mean(lines)),
        "mean_pieces": float(np.mean(pieces)),
        "n_games": len(scores),
    }


def train(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cfg, reward_cfg = build_config(args, device)
    print(
        f"preset={args.preset} device={device} lr={cfg.lr} double_dqn={cfg.double_dqn} "
        f"eps_decay={cfg.epsilon_decay_steps} layer_norm={cfg.use_layer_norm}"
    )

    if args.resume:
        agent = DQNAgent.from_checkpoint(args.resume, device=device, explore=True)
        print(
            f"Resumed from {args.resume} (steps={agent.total_steps}, eps={agent.epsilon:.3f}, "
            f"best_eval={agent.best_eval_score:.1f}, buffer={len(agent.buffer)})"
        )
    else:
        agent = DQNAgent(cfg)

    use_tb = args.tb or not args.no_tb
    writer = None
    if use_tb:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(log_dir / "tb"))
        except Exception as exc:  # pragma: no cover
            print(f"TensorBoard unavailable ({exc}); continuing without it.")

    csv_path = log_dir / "training_metrics.csv"
    write_header = not csv_path.exists()
    csv_file = csv_path.open("a", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "episode",
            "score",
            "lines",
            "pieces",
            "reward",
            "epsilon",
            "loss",
            "steps",
            "terminated",
            "truncated",
        ],
    )
    if write_header:
        csv_writer.writeheader()

    meta = {
        "preset": args.preset,
        "config": cfg.__dict__,
        "reward_config": reward_cfg.__dict__,
        "episodes": args.episodes,
        "seed": args.seed,
        "device": device,
        "validation_seeds": list(VALIDATION_SEEDS),
        "test_seed_base": TEST_SEED_BASE,
    }
    (log_dir / "run_config.json").write_text(json.dumps(meta, indent=2, default=str))

    # Preserve best score across resumed runs.
    best_eval = agent.best_eval_score
    t0 = time.time()
    n_val = min(args.eval_games, len(VALIDATION_SEEDS))

    for ep in range(1, args.episodes + 1):
        env = TetrisEnv(seed=args.seed + ep, reward_config=reward_cfg)
        _, info = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        losses: list[float] = []
        terminated = False
        truncated = False

        while not terminated and not truncated:
            if info["n_legal"] == 0:
                break
            action = agent.select_action(env, explore=True)
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

        agent.episodes_trained += 1
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        row = {
            "episode": ep,
            "score": env.engine.score,
            "lines": env.engine.lines_cleared_total,
            "pieces": env.engine.pieces_placed,
            "reward": ep_reward,
            "epsilon": agent.epsilon,
            "loss": mean_loss,
            "steps": agent.total_steps,
            "terminated": int(terminated),
            "truncated": int(truncated),
        }
        csv_writer.writerow(row)
        csv_file.flush()

        if writer is not None:
            writer.add_scalar("train/score", row["score"], agent.episodes_trained)
            writer.add_scalar("train/lines", row["lines"], agent.episodes_trained)
            writer.add_scalar("train/episode_reward", ep_reward, agent.episodes_trained)
            writer.add_scalar("train/epsilon", agent.epsilon, agent.episodes_trained)
            if losses:
                writer.add_scalar("train/loss", mean_loss, agent.episodes_trained)

        if ep % 10 == 0 or ep == 1:
            elapsed = time.time() - t0
            print(
                f"ep={ep:5d} score={row['score']:5d} lines={row['lines']:3d} "
                f"pieces={row['pieces']:3d} R={ep_reward:7.2f} eps={agent.epsilon:.3f} "
                f"loss={mean_loss:.4f} steps={agent.total_steps} "
                f"T/Tr={int(terminated)}/{int(truncated)} ({elapsed:.1f}s)"
            )

        if ep % args.save_every == 0:
            agent.best_eval_score = best_eval
            agent.save(checkpoint_dir / f"dqn_ep{ep}.pt")
            agent.save(checkpoint_dir / "dqn_latest.pt")

        if args.eval_every > 0 and ep % args.eval_every == 0:
            metrics = evaluate_greedy(agent, args.max_pieces, n_games=n_val)
            print(
                f"  [val n={metrics['n_games']}] mean_score={metrics['mean_score']:.1f} "
                f"mean_lines={metrics['mean_lines']:.2f} "
                f"mean_pieces={metrics['mean_pieces']:.1f} "
                f"(best={best_eval:.1f})"
            )
            if writer is not None:
                writer.add_scalar("val/mean_score", metrics["mean_score"], agent.episodes_trained)
                writer.add_scalar("val/mean_lines", metrics["mean_lines"], agent.episodes_trained)
            if metrics["mean_score"] >= best_eval:
                best_eval = metrics["mean_score"]
                agent.best_eval_score = best_eval
                agent.save(checkpoint_dir / "dqn_best.pt")

    agent.best_eval_score = best_eval
    agent.save(checkpoint_dir / "dqn_final.pt")
    agent.save(checkpoint_dir / "dqn_latest.pt")
    csv_file.close()
    if writer is not None:
        writer.close()
    print(f"Training complete. Checkpoints in {checkpoint_dir} (best_eval={best_eval:.1f})")
    return checkpoint_dir / "dqn_final.pt"


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    train(args)


if __name__ == "__main__":
    main()
