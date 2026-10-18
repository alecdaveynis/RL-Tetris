"""
Compare random, heuristic, and DQN agents on the same seeded game set.

Saves a combined CSV and comparison charts (matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.evaluation.evaluate import AgentProtocol, evaluate_agent, save_results


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare Tetris agents")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pieces", type=int, default=500)
    p.add_argument("--model", type=str, default="models/dqn_best.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="results/comparison")
    p.add_argument("--skip-dqn", action="store_true", help="Compare only baselines")
    return p.parse_args(argv)


def plot_comparisons(
    agent_to_scores: dict[str, list[int]],
    agent_to_lines: dict[str, list[int]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(agent_to_scores.keys())

    # Bar chart: mean score ± std
    means = [float(np.mean(agent_to_scores[n])) for n in names]
    stds = [float(np.std(agent_to_scores[n], ddof=1)) if len(agent_to_scores[n]) > 1 else 0.0 for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=6, color=["#4C72B0", "#55A868", "#C44E52"][: len(names)])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Score")
    ax.set_title("Mean game score (±1 std)")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_scores.png", dpi=150)
    plt.close(fig)

    # Box plot of scores
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([agent_to_scores[n] for n in names], tick_labels=names)
    ax.set_ylabel("Score")
    ax.set_title("Score distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "score_boxplot.png", dpi=150)
    plt.close(fig)

    # Mean lines
    line_means = [float(np.mean(agent_to_lines[n])) for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, line_means, color=["#4C72B0", "#55A868", "#C44E52"][: len(names)])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Lines cleared")
    ax.set_title("Mean lines cleared")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_lines.png", dpi=150)
    plt.close(fig)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agents: list[AgentProtocol] = [
        RandomAgent(seed=args.seed),
        HeuristicAgent(seed=args.seed),
    ]
    if not args.skip_dqn:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(
                f"DQN model not found at {model_path}. Train first or pass --skip-dqn."
            )
        agents.append(DQNAgent.from_checkpoint(model_path, device=args.device, explore=False))

    summaries = []
    agent_scores: dict[str, list[int]] = {}
    agent_lines: dict[str, list[int]] = {}

    for agent in agents:
        print(f"Evaluating {agent.name()} on {args.games} games...")
        summary, results = evaluate_agent(
            agent, n_games=args.games, seed=args.seed, max_pieces=args.max_pieces
        )
        save_results(summary, results, out_dir)
        summaries.append(summary)
        agent_scores[agent.name()] = [r.score for r in results]
        agent_lines[agent.name()] = [r.lines_cleared for r in results]
        print(
            f"  mean_score={summary.mean_score:.2f} median={summary.median_score:.2f} "
            f"mean_lines={summary.mean_lines:.2f} top_out_rate={summary.top_out_rate:.2f}"
        )

    # Combined CSV
    combined = out_dir / "comparison_summary.csv"
    with combined.open("w", newline="") as f:
        fieldnames = list(asdict(summaries[0]).keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))

    (out_dir / "comparison_summary.json").write_text(
        json.dumps([asdict(s) for s in summaries], indent=2)
    )
    plot_comparisons(agent_scores, agent_lines, out_dir)
    print(f"Comparison saved to {out_dir}")


if __name__ == "__main__":
    main()
