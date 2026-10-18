"""Random baseline: uniform selection among legal placements."""

from __future__ import annotations

from typing import Optional

import numpy as np

from tetris_rl.environment import TetrisEnv


class RandomAgent:
    """Selects a legal action uniformly at random."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, env: TetrisEnv, explore: bool = False) -> int:
        n = len(env.legal_placements())
        if n == 0:
            raise RuntimeError("No legal actions")
        return int(self.rng.integers(0, n))

    def name(self) -> str:
        return "random"
