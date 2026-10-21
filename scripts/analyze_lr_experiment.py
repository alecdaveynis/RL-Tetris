#!/usr/bin/env python3
"""Post-hoc LR experiment analysis: val curves, Q/TD diagnostics, held-out test."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.environment import TetrisEnv
from tetris_rl.evaluation.evaluate import evaluate_agent, save_results
from tetris_rl.features import placement_feature_batch
from tetris_rl.training.train_dqn import TEST_SEED_BASE, VALIDATION_SEEDS, evaluate_greedy

FROZEN = Path("models/frozen/dqn_best_val1320.pt")
DIAG_SEEDS = tuple(VALIDATION_SEEDS[:8])
VAL_N = 50
TEST_N = 100
MAX_PIECES = 500
OUT = Path("results/exp_lr_compare")


def param_rms(a: nn.Module, b: nn.Module) -> float:
    tot = 0.0
    n = 0
    for p, q in zip(a.parameters(), b.parameters()):
        tot += float((p.detach() - q.detach()).pow(2).sum().cpu())
        n += p.numel()
    return (tot / max(n, 1)) ** 0.5


def fixed_state_q(agent: DQNAgent) -> dict:
    qs_mean, qs_max = [], []
    for seed in DIAG_SEEDS:
        env = TetrisEnv(seed=int(seed))
        env.reset(seed=int(seed))
        feats = placement_feature_batch(env.engine, env.legal_placements())
        q = agent.q_values(feats, agent.online)
        qs_mean.append(float(np.mean(q)))
        qs_max.append(float(np.max(q)))
    return {
        "diag_q_mean": float(np.mean(qs_mean)),
        "diag_q_max": float(np.mean(qs_max)),
        "diag_q_max_abs": float(np.max(qs_max)),
    }


def replay_td(agent: DQNAgent) -> dict:
    n = min(64, len(agent.buffer))
    batch = agent.buffer.sample(n)
    sa = torch.as_tensor(
        np.stack([t.state_action for t in batch], axis=0),
        dtype=torch.float32,
        device=agent.device,
    )
    agent.online.eval()
    with torch.no_grad():
        q = agent.online(sa)
        targets = agent.bellman_targets(batch, vectorized=True)
        td = targets - q
        huber = float(torch.nn.functional.smooth_l1_loss(q, targets).cpu())
    agent.optimizer.zero_grad()
    loss = agent._compute_loss(batch)
    loss.backward()
    g2 = 0.0
    for p in agent.online.parameters():
        if p.grad is not None:
            g2 += float(p.grad.detach().pow(2).sum().cpu())
    agent.optimizer.zero_grad()
    return {
        "huber": huber,
        "td_abs_mean": float(td.abs().mean().cpu()),
        "q_mean_replay": float(q.mean().cpu()),
        "q_max_replay": float(q.max().cpu()),
        "grad_norm": g2 ** 0.5,
        "online_target_rms": param_rms(agent.online, agent.target),
        "optimizer_lr": float(agent.optimizer.param_groups[0]["lr"]),
        "buffer_size": len(agent.buffer),
        "grad_steps": agent.grad_steps,
    }


def snapshot(path: Path, extra_episodes: int, device: str) -> dict:
    agent = DQNAgent.from_checkpoint(path, device=device, explore=False)
    val = evaluate_greedy(agent, MAX_PIECES, n_games=VAL_N)
    row = {
        "path": str(path),
        "extra_episodes": extra_episodes,
        "steps": agent.total_steps,
        "episodes_trained": agent.episodes_trained,
        **val,
        **fixed_state_q(agent),
        **replay_td(agent),
    }
    return row


def parse_train_loss(log_dir: Path) -> list[dict]:
    csv_path = log_dir / "training_metrics.csv"
    if not csv_path.exists():
        return []
    rows = list(csv.DictReader(csv_path.open()))
    out = []
    for r in rows:
        out.append(
            {
                "episode": int(r["episode"]),
                "loss": float(r["loss"]) if r["loss"] not in ("nan", "") else None,
                "score": int(r["score"]),
            }
        )
    return out


def plot_curves(series: dict[str, list[dict]], losses: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def xy(rows, key):
        return [r["extra_episodes"] for r in rows], [r[key] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in series.items():
        x, y = xy(rows, "mean_score")
        ax.plot(x, y, marker="o", label=name)
    ax.set_xlabel("Additional training episodes")
    ax.set_ylabel("Mean validation score")
    ax.set_title("Validation score vs additional episodes (50 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "val_score.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in losses.items():
        xs = [r["episode"] for r in rows if r["loss"] is not None]
        ys = [r["loss"] for r in rows if r["loss"] is not None]
        ax.plot(xs, ys, alpha=0.85, label=name)
    ax.set_xlabel("Training episode in run")
    ax.set_ylabel("Huber loss")
    ax.set_title("Training loss vs episodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in series.items():
        x, y = xy(rows, "diag_q_mean")
        ax.plot(x, y, marker="o", label=name)
    ax.set_xlabel("Additional training episodes")
    ax.set_ylabel("Mean Q (fixed start states)")
    ax.set_title("Mean Q-value vs additional episodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mean_q.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, rows in series.items():
        x, y = xy(rows, "mean_lines")
        ax.plot(x, y, marker="o", label=name)
    ax.set_xlabel("Additional training episodes")
    ax.set_ylabel("Mean lines cleared")
    ax.set_title("Mean lines vs additional episodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "mean_lines.png", dpi=150)
    plt.close(fig)


def pick_best(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (r["mean_score"], r["median_score"]))


def main() -> None:
    device = "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    frozen_row = snapshot(FROZEN, 0, device)
    print("frozen val", frozen_row["mean_score"], "median", frozen_row["median_score"])

    runs = {
        "lr_5e-4": Path("models/exp_lr_5e4_200"),
        "lr_1e-4": Path("models/exp_lr_1e4_200"),
    }
    logs = {
        "lr_5e-4": Path("results/exp_lr_5e4_200"),
        "lr_1e-4": Path("results/exp_lr_1e4_200"),
    }

    series: dict[str, list[dict]] = {}
    for name, ckpt_dir in runs.items():
        rows = [dict(frozen_row, run=name)]
        for ep in (50, 100, 150, 200):
            p = ckpt_dir / f"dqn_ep{ep}.pt"
            if not p.exists():
                p = ckpt_dir / "dqn_final.pt" if ep == 200 else None
            if p is None or not p.exists():
                print("missing", ckpt_dir, ep)
                continue
            row = snapshot(p, ep, device)
            row["run"] = name
            rows.append(row)
            print(name, "ep", ep, "val_mean", row["mean_score"], "q", row["diag_q_mean"], "huber", row["huber"])
        series[name] = rows

    losses = {name: parse_train_loss(logs[name]) for name in runs}
    plot_curves(series, losses, OUT)

    (OUT / "snapshots.json").write_text(json.dumps({"frozen": frozen_row, "series": series}, indent=2))

    # Select best-of-run by validation only (includes episode 0 = frozen).
    selections = {}
    for name, rows in series.items():
        best = pick_best(rows)
        final = next((r for r in rows if r["extra_episodes"] == 200), rows[-1])
        selections[name] = {"best": best, "final": final}
        print(
            f"SELECT {name}: best extra_ep={best['extra_episodes']} val={best['mean_score']:.1f} "
            f"path={best['path']}"
        )

    # Held-out test only after val selection.
    test_agents = {
        "frozen": FROZEN,
        "lr_5e-4_best": Path(selections["lr_5e-4"]["best"]["path"]),
        "lr_1e-4_best": Path(selections["lr_1e-4"]["best"]["path"]),
        "lr_5e-4_final": Path(selections["lr_5e-4"]["final"]["path"]),
        "lr_1e-4_final": Path(selections["lr_1e-4"]["final"]["path"]),
    }
    # Deduplicate identical paths
    tested = {}
    summaries = []
    for label, path in test_agents.items():
        key = str(path.resolve())
        if key in tested:
            print("test skip duplicate", label, path)
            summaries.append({**tested[key], "agent": label})
            continue
        from tetris_rl.agents.dqn_agent import DQNAgent as A
        from tetris_rl.evaluation.compare_ab import NamedAgent

        inner = A.from_checkpoint(path, device=device, explore=False)
        agent = NamedAgent(inner, label)
        summary, results = evaluate_agent(agent, n_games=TEST_N, seed=TEST_SEED_BASE, max_pieces=MAX_PIECES)
        tested[key] = summary.__dict__ if hasattr(summary, "__dict__") else dict(summary)
        # dataclass
        from dataclasses import asdict

        tested[key] = asdict(summary)
        summaries.append(asdict(summary) | {"label": label, "path": str(path)})
        save_results(summary, results, OUT / "test")
        print("TEST", label, "mean", summary.mean_score, "median", summary.median_score)

    (OUT / "test_summaries.json").write_text(json.dumps(summaries, indent=2))
    (OUT / "selections.json").write_text(json.dumps(selections, indent=2, default=str))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
