"""
Gymnasium environment wrapping the Tetris engine with placement-level actions.

Action space is a fixed Discrete(max_actions) with a legal-action mask.
Invalid indices must not be selected by agents; the env raises on illegal acts.

Stable-Baselines3's standard DQN does not support action masking. This project
implements a custom PyTorch placement-action DQN instead (see agents/dqn_agent.py).
"""

from __future__ import annotations

from typing import Any, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tetris_rl.engine import BOARD_HEIGHT, BOARD_WIDTH, Placement, TetrisEngine
from tetris_rl.features import observation_size, observation_vector
from tetris_rl.rewards import DEFAULT_REWARD_CONFIG, RewardConfig, shaped_reward

# Max distinct placements: 4 rotations × 10 columns (with some margin for offsets).
MAX_ACTIONS = 40


class TetrisEnv(gym.Env):
    """Placement-level Tetris Gymnasium environment."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        seed: Optional[int] = None,
        reward_config: Optional[RewardConfig] = None,
        max_actions: int = MAX_ACTIONS,
    ) -> None:
        super().__init__()
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self.max_actions = max_actions
        self.engine = TetrisEngine(seed=seed)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size(),),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(max_actions)

        self._legal: list[Placement] = []
        self._last_lines: int = 0
        self._episode_steps: int = 0

    def legal_placements(self) -> list[Placement]:
        return list(self._legal)

    def action_mask(self) -> np.ndarray:
        """Boolean mask of shape (max_actions,) — True where action is legal."""
        mask = np.zeros(self.max_actions, dtype=bool)
        n = min(len(self._legal), self.max_actions)
        mask[:n] = True
        return mask

    def _refresh_legal(self) -> None:
        self._legal = self.engine.enumerate_placements()
        if len(self._legal) > self.max_actions:
            # Extremely unlikely with standard board; truncate deterministically.
            self._legal = self._legal[: self.max_actions]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.engine.reset(seed=seed)
        self._last_lines = 0
        self._episode_steps = 0
        self._refresh_legal()
        if not self._legal:
            self.engine.game_over = True
        obs = observation_vector(self.engine, lines_just_cleared=0)
        return obs, self._info()

    def step(self, action: int) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.engine.game_over:
            raise RuntimeError("Episode is done; call reset()")

        if action < 0 or action >= len(self._legal):
            raise ValueError(
                f"Illegal action {action}; only {len(self._legal)} legal placements. "
                "Use action_mask() or legal_placements()."
            )

        placement = self._legal[action]
        board_before = self.engine.copy_board()

        result = self.engine.step(placement)
        board_after = self.engine.copy_board()
        self._last_lines = result.lines_cleared
        self._episode_steps += 1

        reward = shaped_reward(
            self.engine,
            board_before,
            board_after,
            result.lines_cleared,
            result.game_over,
            config=self.reward_config,
        )

        if not result.game_over:
            self._refresh_legal()
            if not self._legal:
                self.engine.game_over = True
                reward -= self.reward_config.game_over_penalty
        else:
            self._legal = []

        terminated = self.engine.game_over
        truncated = False
        obs = observation_vector(
            self.engine,
            lines_just_cleared=result.lines_cleared if not terminated else 0,
        )
        info = self._info()
        info["score_gained"] = result.score_gained
        info["lines_cleared"] = result.lines_cleared
        info["placement"] = placement
        return obs, reward, terminated, truncated, info

    def _info(self) -> dict[str, Any]:
        return {
            "score": self.engine.score,
            "lines_cleared_total": self.engine.lines_cleared_total,
            "pieces_placed": self.engine.pieces_placed,
            "game_over": self.engine.game_over,
            "action_mask": self.action_mask(),
            "n_legal": len(self._legal),
            "current_piece": int(self.engine.current_piece),
            "next_piece": int(self.engine.next_piece),
            "episode_steps": self._episode_steps,
        }

    def render(self) -> Optional[str]:
        rows = []
        for r in range(BOARD_HEIGHT):
            row = "".join("#" if self.engine.board[r, c] else "." for c in range(BOARD_WIDTH))
            rows.append(row)
        return "\n".join(rows)


def make_env(seed: Optional[int] = None, **kwargs: Any) -> TetrisEnv:
    return TetrisEnv(seed=seed, **kwargs)
