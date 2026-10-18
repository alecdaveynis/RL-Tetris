#!/usr/bin/env python3
"""Benchmark loop vs vectorized Bellman / next-Q computation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from tetris_rl.agents.dqn_agent import DQNAgent, DQNConfig
from tetris_rl.environment import TetrisEnv


def fill(agent: DQNAgent, n: int = 200, seed: int = 0) -> None:
    env = TetrisEnv(seed=seed)
    env.reset(seed=seed)
    for _ in range(n):
        if env.engine.game_over or not env.legal_placements():
            env.reset()
        action = agent.select_action(env, explore=True)
        sa = agent.features_for_action(env, action)
        _, reward, terminated, truncated, _ = env.step(action)
        agent.observe(sa, float(reward), env, terminated=terminated, truncated=truncated)
        if terminated or truncated:
            env.reset()


def bench(agent: DQNAgent, batch_size: int, repeats: int, vectorized: bool) -> float:
    # Warmup
    batch = agent.buffer.sample(batch_size)
    for _ in range(3):
        _ = agent.bellman_targets(batch, vectorized=vectorized)
    if agent.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        batch = agent.buffer.sample(batch_size)
        _ = agent.bellman_targets(batch, vectorized=vectorized)
    if agent.device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--buffer-fill", type=int, default=400)
    p.add_argument("--out", type=str, default="results/bench_dqn_loss.json")
    args = p.parse_args()

    cfg = DQNConfig(
        device=args.device,
        seed=0,
        batch_size=args.batch_size,
        buffer_size=5_000,
        min_buffer_size=args.batch_size,
        double_dqn=True,
        hidden_sizes=(256, 256, 128),
        use_layer_norm=True,
    )
    agent = DQNAgent(cfg)
    fill(agent, n=args.buffer_fill, seed=0)
    assert len(agent.buffer) >= args.batch_size

    # Equivalence check on one batch
    batch = agent.buffer.sample(args.batch_size)
    with torch.no_grad():
        tv = agent.bellman_targets(batch, vectorized=True)
        tl = agent.bellman_targets(batch, vectorized=False)
    max_err = float((tv - tl).abs().max().item())

    t_loop = bench(agent, args.batch_size, args.repeats, vectorized=False)
    t_vec = bench(agent, args.batch_size, args.repeats, vectorized=True)

    result = {
        "device": args.device,
        "batch_size": args.batch_size,
        "repeats": args.repeats,
        "buffer_size": len(agent.buffer),
        "max_abs_error_vec_vs_loop": max_err,
        "seconds_loop": t_loop,
        "seconds_vectorized": t_vec,
        "speedup_x": (t_loop / t_vec) if t_vec > 0 else None,
        "ms_per_call_loop": 1000.0 * t_loop / args.repeats,
        "ms_per_call_vectorized": 1000.0 * t_vec / args.repeats,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
