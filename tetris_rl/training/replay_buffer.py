"""Experience replay buffer for placement-action DQN transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class Transition:
    """One placement decision transition.

    ``terminated`` is True only for true game-over (no Bellman bootstrap).
    Time-limit truncations keep ``terminated=False`` and store next-action
    features so the bootstrap target remains valid.
    """

    state_action: np.ndarray  # features of chosen (s, a)
    reward: float
    next_state_actions: np.ndarray  # (N', F) features for legal next actions
    terminated: bool

    @property
    def done(self) -> bool:
        """Alias for ``terminated`` (older call sites / checkpoints)."""
        return self.terminated


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
        terminated: bool = False,
        *,
        done: Optional[bool] = None,
    ) -> None:
        """Add a transition. ``done`` is accepted as an alias for ``terminated``."""
        if done is not None:
            terminated = bool(done)
        self._storage[self._idx] = Transition(
            state_action=np.asarray(state_action, dtype=np.float32),
            reward=float(reward),
            next_state_actions=np.asarray(next_state_actions, dtype=np.float32),
            terminated=bool(terminated),
        )
        self._idx = (self._idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> list[Transition]:
        if batch_size > self._size:
            raise ValueError(f"Cannot sample {batch_size} from buffer of size {self._size}")
        # When not full, filled slots are [0, size). When full, all physical
        # slots are valid — sample indices in [0, size) as physical addresses.
        indices = self.rng.choice(self._size, size=batch_size, replace=False)
        return [self._storage[i] for i in indices]  # type: ignore[misc]

    def can_sample(self, batch_size: int) -> bool:
        return self._size >= batch_size

    @staticmethod
    def _encode_transition(t: Transition) -> dict[str, Any]:
        return {
            "state_action": t.state_action,
            "reward": t.reward,
            "next_state_actions": t.next_state_actions,
            "terminated": t.terminated,
        }

    @staticmethod
    def _decode_transition(item: dict[str, Any]) -> Transition:
        terminated = bool(item.get("terminated", item.get("done", False)))
        return Transition(
            state_action=np.asarray(item["state_action"], dtype=np.float32),
            reward=float(item["reward"]),
            next_state_actions=np.asarray(item["next_state_actions"], dtype=np.float32),
            terminated=terminated,
        )

    def state_dict(self) -> dict[str, Any]:
        """Serialize buffer contents and RNG for checkpointing."""
        storage: list[Optional[dict[str, Any]]] = []
        for t in self._storage:
            storage.append(None if t is None else self._encode_transition(t))
        return {
            "capacity": self.capacity,
            "idx": self._idx,
            "size": self._size,
            "rng_state": self.rng.bit_generator.state,
            "storage": storage,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore buffer from ``state_dict`` (tolerant of older formats)."""
        capacity = int(state.get("capacity", self.capacity))
        self.capacity = capacity
        self._idx = int(state.get("idx", 0))
        self._size = int(state.get("size", 0))
        if "rng_state" in state and state["rng_state"] is not None:
            self.rng.bit_generator.state = state["rng_state"]

        if "storage" in state:
            raw = state["storage"]
            self._storage = [None] * capacity
            for i, item in enumerate(raw[:capacity]):
                if item is not None:
                    self._storage[i] = self._decode_transition(item)
            return

        # Older "transitions" list (oldest→newest logical order).
        raw = state.get("transitions", [])
        self._storage = [None] * capacity
        self._idx = 0
        self._size = 0
        for item in raw:
            if item is None:
                continue
            t = self._decode_transition(item)
            self.add(t.state_action, t.reward, t.next_state_actions, t.terminated)
