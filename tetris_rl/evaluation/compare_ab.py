"""
Compare random, heuristic, original DQN, and improved DQN on shared seeds.
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


class NamedAgent:
    """Wrap an agent with a custom display name for comparison tables."""

    def __init__(self, agent: AgentProtocol, label: str) -> None:
        self._agent = agent
        self._label = label

    def select_action(self, env, explore: bool = False) -> int:
        return self._agent.select_action(env, explore=explore)

    def name(self) -> str:
        return self._label


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B agent comparison")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pieces", type=int, default=500)
    p.add_argument("--original-model", type=str, default="models/original/dqn_best.pt")
    p.add_argument("--improved-model", type=str, default="models/improved/dqn_best.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="results/ab_comparison")
    p.add_argument("--skip-dqn", action="store_true")
    return p.parse_args(argv)


def plot_comparisons(
    agent_scores: dict[str, list[int]],
    agent_lines: dict[str, list[int]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(agent_scores.keys())
    x = np.arange(len(names))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    means = [float(np.mean(agent_scores[n])) for n in names]
    stds = [
        float(np.std(agent_scores[n], ddof=1)) if len(agent_scores[n]) > 1 else 0.0
        for n in names
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, means, yerr=stds, capsize=5, color=colors[: len(names)])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Mean game score (±1 std)")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_scores.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot([agent_scores[n] for n in names], tick_labels=names)
    ax.set_ylabel("Score")
    ax.set_title("Score distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "score_boxplot.png", dpi=150)
    plt.close(fig)

    line_means = [float(np.mean(agent_lines[n])) for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, line_means, color=colors[: len(names)])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
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
        NamedAgent(RandomAgent(seed=args.seed), "random"),
        NamedAgent(HeuristicAgent(seed=args.seed), "heuristic"),
    ]
    if not args.skip_dqn:
        for label, path in [
            ("dqn_original", args.original_model),
            ("dqn_improved", args.improved_model),
        ]:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Missing model for {label}: {p}")
            agents.append(
                NamedAgent(
                    DQNAgent.from_checkpoint(p, device=args.device, explore=False),
                    label,
                )
            )

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
            f"best={summary.best_score:.0f} mean_lines={summary.mean_lines:.2f} "
            f"mean_pieces={summary.mean_pieces:.1f} std_score={summary.std_score:.2f}"
        )

    with (out_dir / "comparison_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(summaries[0]).keys()))
        writer.writeheader()
        for s in summaries:
            writer.writerow(asdict(s))

    (out_dir / "comparison_summary.json").write_text(
        json.dumps([asdict(s) for s in summaries], indent=2)
    )
    plot_comparisons(agent_scores, agent_lines, out_dir)
    print(f"Saved comparison to {out_dir}")


if __name__ == "__main__":
    main()
