"""
Evaluate a single agent over seeded games and report summary statistics.

Evaluation metrics use true game score and lines cleared — not shaped rewards.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from tetris_rl.agents.dqn_agent import DQNAgent
from tetris_rl.agents.heuristic_agent import HeuristicAgent
from tetris_rl.agents.random_agent import RandomAgent
from tetris_rl.environment import TetrisEnv


class AgentProtocol(Protocol):
    def select_action(self, env: TetrisEnv, explore: bool = False) -> int: ...

    def name(self) -> str: ...


@dataclass
class GameResult:
    seed: int
    score: int
    lines_cleared: int
    pieces_placed: int
    topped_out: bool
    survival_steps: int


@dataclass
class EvalSummary:
    agent: str
    n_games: int
    mean_score: float
    median_score: float
    std_score: float
    best_score: float
    mean_lines: float
    median_lines: float
    std_lines: float
    mean_pieces: float
    median_pieces: float
    mean_survival: float
    top_out_rate: float


def run_episode(
    agent: AgentProtocol,
    seed: int,
    max_pieces: int = 500,
) -> GameResult:
    env = TetrisEnv(seed=seed)
    _, info = env.reset(seed=seed)
    done = False
    # Force greedy for learned agents
    explore = False
    while not done and env.engine.pieces_placed < max_pieces:
        if info["n_legal"] == 0:
            break
        action = agent.select_action(env, explore=explore)
        _, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return GameResult(
        seed=seed,
        score=env.engine.score,
        lines_cleared=env.engine.lines_cleared_total,
        pieces_placed=env.engine.pieces_placed,
        topped_out=env.engine.game_over,
        survival_steps=env.engine.pieces_placed,
    )


def summarize(agent_name: str, results: list[GameResult]) -> EvalSummary:
    scores = np.array([r.score for r in results], dtype=np.float64)
    lines = np.array([r.lines_cleared for r in results], dtype=np.float64)
    pieces = np.array([r.pieces_placed for r in results], dtype=np.float64)
    survival = np.array([r.survival_steps for r in results], dtype=np.float64)
    top_outs = np.array([r.topped_out for r in results], dtype=np.float64)
    return EvalSummary(
        agent=agent_name,
        n_games=len(results),
        mean_score=float(np.mean(scores)),
        median_score=float(np.median(scores)),
        std_score=float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        best_score=float(np.max(scores)),
        mean_lines=float(np.mean(lines)),
        median_lines=float(np.median(lines)),
        std_lines=float(np.std(lines, ddof=1)) if len(lines) > 1 else 0.0,
        mean_pieces=float(np.mean(pieces)),
        median_pieces=float(np.median(pieces)),
        mean_survival=float(np.mean(survival)),
        top_out_rate=float(np.mean(top_outs)),
    )


def evaluate_agent(
    agent: AgentProtocol,
    n_games: int = 100,
    seed: int = 0,
    max_pieces: int = 500,
) -> tuple[EvalSummary, list[GameResult]]:
    results = [run_episode(agent, seed=seed + i, max_pieces=max_pieces) for i in range(n_games)]
    return summarize(agent.name(), results), results


def load_agent(name: str, model_path: Optional[str] = None, device: str = "cpu") -> AgentProtocol:
    name = name.lower()
    if name == "random":
        return RandomAgent(seed=0)
    if name == "heuristic":
        return HeuristicAgent(seed=0)
    if name == "dqn":
        if not model_path:
            raise ValueError("--model is required for dqn agent")
        return DQNAgent.from_checkpoint(model_path, device=device, explore=False)
    raise ValueError(f"Unknown agent: {name}")


def save_results(
    summary: EvalSummary,
    results: list[GameResult],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{summary.agent}_summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2))

    games_path = out_dir / f"{summary.agent}_games.csv"
    with games_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "score",
                "lines_cleared",
                "pieces_placed",
                "topped_out",
                "survival_steps",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a Tetris agent")
    p.add_argument("--agent", type=str, required=True, choices=["random", "heuristic", "dqn"])
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-pieces", type=int, default=500)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out-dir", type=str, default="results/eval")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    agent = load_agent(args.agent, args.model, args.device)
    summary, results = evaluate_agent(agent, args.games, args.seed, args.max_pieces)
    out = Path(args.out_dir)
    save_results(summary, results, out)
    print(json.dumps(asdict(summary), indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
