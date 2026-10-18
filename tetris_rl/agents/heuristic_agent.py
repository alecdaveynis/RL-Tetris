"""
Heuristic baseline using Dellacherie-style weighted board evaluation.

Evaluates each legal placement by simulating the resulting board and scoring
a linear combination of lines cleared, aggregate height, holes, and bumpiness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tetris_rl.environment import TetrisEnv
from tetris_rl.features import board_metrics


@dataclass
class HeuristicWeights:
    lines_cleared: float = 0.76
    aggregate_height: float = -0.51
    holes: float = -0.36
    bumpiness: float = -0.18


DEFAULT_WEIGHTS = HeuristicWeights()


class HeuristicAgent:
    """Greedy placement agent with configurable heuristic weights."""

    def __init__(
        self,
        weights: Optional[HeuristicWeights] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.weights = weights or DEFAULT_WEIGHTS
        self.rng = np.random.default_rng(seed)

    def evaluate_placement(self, env: TetrisEnv, action_idx: int) -> float:
        placement = env.legal_placements()[action_idx]
        new_board, lines, _ = env.engine.simulate_placement(placement)
        m = board_metrics(new_board)
        w = self.weights
        return (
            w.lines_cleared * lines
            + w.aggregate_height * m["aggregate_height"]
            + w.holes * m["holes"]
            + w.bumpiness * m["bumpiness"]
        )

    def select_action(self, env: TetrisEnv, explore: bool = False) -> int:
        n = len(env.legal_placements())
        if n == 0:
            raise RuntimeError("No legal actions")
        scores = np.array([self.evaluate_placement(env, i) for i in range(n)], dtype=np.float64)
        best = float(np.max(scores))
        # Break ties randomly for diversity under identical boards.
        candidates = np.flatnonzero(np.isclose(scores, best))
        return int(self.rng.choice(candidates))

    def name(self) -> str:
        return "heuristic"
