"""Experience replay buffer for placement-action DQN transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Transition:
    """One placement decision transition."""

    state_action: np.ndarray  # features of chosen (s, a)
    reward: float
    next_state_actions: np.ndarray  # (N', F) features for all legal next actions
    done: bool


class ReplayBuffer:
    """Fixed-capacity circular buffer storing variable next-action sets."""

    def __init__(self, capacity: int = 50_000, seed: Optional[int] = None) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self._storage: list[Optional[Transition]] = [None] * capacity
        self._idx = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state_action: np.ndarray,
        reward: float,
        next_state_actions: np.ndarray,
        done: bool,
    ) -> None:
        self._storage[self._idx] = Transition(
            state_action=np.asarray(state_action, dtype=np.float32),
            reward=float(reward),
            next_state_actions=np.asarray(next_state_actions, dtype=np.float32),
            done=bool(done),
        )
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > self._size:
            raise ValueError(f"Cannot sample {batch_size} from buffer of size {self._size}")
        indices = self.rng.choice(self._size, size=batch_size, replace=False)
        return [self._storage[i] for i in indices]  # type: ignore[misc]

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size
