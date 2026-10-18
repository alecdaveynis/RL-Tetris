"""
Placement-action Deep Q-Network for Tetris.

Stable-Baselines3's DQN expects a fixed Discrete action space and does not
support action masking. Instead we train a Q-network that scores each legal
placement from its state-action feature vector:

    Q(s, a) = f_θ(φ(s, a))

At decision time we enumerate legal placements, compute φ for each, and pick
argmax_a Q(s, a) (ε-greedy during training). Bellman targets bootstrap from
max over legal next actions (or Double DQN: online argmax, target evaluation).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tetris_rl.environment import TetrisEnv
from tetris_rl.features import (
    placement_feature_batch,
    state_action_feature_size,
    state_action_features,
)
from tetris_rl.training.replay_buffer import ReplayBuffer, Transition


@dataclass
class DQNConfig:
    """Hyperparameters for placement-action DQN."""

    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer_size: int = 1_000
    target_update_freq: int = 500  # gradient steps between hard target copies
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    hidden_sizes: tuple[int, ...] = (256, 256, 128)
    grad_clip: float = 10.0
    double_dqn: bool = False
    use_layer_norm: bool = False
    device: str = "cpu"
    seed: Optional[int] = None


def original_dqn_config(**overrides: Any) -> DQNConfig:
    """Pre-audit defaults (config A)."""
    cfg = DQNConfig(
        lr=1e-3,
        double_dqn=False,
        epsilon_decay_steps=20_000,
        target_update_freq=500,
        grad_clip=10.0,
        use_layer_norm=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def improved_dqn_config(**overrides: Any) -> DQNConfig:
    """
    Post-audit defaults (config B): Double DQN, moderate LR, delta-friendly
    exploration horizon, more frequent target sync, input LayerNorm.

    Note: epsilon_decay_steps=15_000 reaches ε≈0.05 by ~15k env steps so that
    short equal-budget runs (e.g. 400 episodes) still leave time for greedy
    learning. Increase for multi-thousand-episode training.
    """
    cfg = DQNConfig(
        lr=5e-4,
        double_dqn=True,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=15_000,
        target_update_freq=250,
        grad_clip=5.0,
        use_layer_norm=True,
        batch_size=64,
        buffer_size=50_000,
        min_buffer_size=1_000,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class QNetwork(nn.Module):
    """MLP scoring a single state-action feature vector → Q-value."""

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256, 128),
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.input_norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values with shape (batch,) from features (batch, F)."""
        return self.net(self.input_norm(x)).squeeze(-1)


class DQNAgent:
    """ε-greedy placement-action DQN with replay and target network."""

    def __init__(self, config: Optional[DQNConfig] = None) -> None:
        self.config = config or DQNConfig()
        self.device = torch.device(self.config.device)
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)

        self.feature_dim = state_action_feature_size()
        self.online = QNetwork(
            self.feature_dim,
            self.config.hidden_sizes,
            use_layer_norm=self.config.use_layer_norm,
        ).to(self.device)
        self.target = QNetwork(
            self.feature_dim,
            self.config.hidden_sizes,
            use_layer_norm=self.config.use_layer_norm,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.lr)
        self.buffer = ReplayBuffer(self.config.buffer_size, seed=self.config.seed)
        self.rng = np.random.default_rng(self.config.seed)

        self.epsilon = self.config.epsilon_start
        self.total_steps = 0
        self.grad_steps = 0
        self.learn_steps = 0

    def name(self) -> str:
        return "dqn"

    def _decay_epsilon(self) -> None:
        cfg = self.config
        frac = min(1.0, self.total_steps / max(1, cfg.epsilon_decay_steps))
        self.epsilon = cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    @torch.no_grad()
    def q_values(self, features: np.ndarray, network: Optional[QNetwork] = None) -> np.ndarray:
        net = network or self.online
        if features.size == 0:
            return np.zeros(0, dtype=np.float32)
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        net.eval()
        return net(x).cpu().numpy()

    def select_action(self, env: TetrisEnv, explore: bool = True) -> int:
        placements = env.legal_placements()
        n = len(placements)
        if n == 0:
            raise RuntimeError("No legal actions")

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, n))

        feats = placement_feature_batch(env.engine, placements)
        q = self.q_values(feats, self.online)
        return int(np.argmax(q))

    def observe(
        self,
        env_before_features: np.ndarray,
        reward: float,
        env: TetrisEnv,
        done: bool,
    ) -> None:
        """Store transition after env.step; next features from current legal set."""
        if done:
            next_feats = np.zeros((0, self.feature_dim), dtype=np.float32)
        else:
            next_feats = placement_feature_batch(env.engine, env.legal_placements())
        self.buffer.add(env_before_features, reward, next_feats, done)
        self.total_steps += 1
        self._decay_epsilon()

    def features_for_action(self, env: TetrisEnv, action: int) -> np.ndarray:
        return state_action_features(env.engine, env.legal_placements()[action])

    def train_step(self) -> Optional[float]:
        """One gradient update. Returns loss, or None if buffer too small."""
        cfg = self.config
        if not self.buffer.can_sample(cfg.batch_size):
            return None

        batch = self.buffer.sample(cfg.batch_size)
        loss = self._compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.optimizer.step()

        self.grad_steps += 1
        self.learn_steps += 1
        if self.grad_steps % cfg.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def _compute_loss(self, batch: list[Transition]) -> torch.Tensor:
        cfg = self.config
        sa = torch.as_tensor(
            np.stack([t.state_action for t in batch], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            [t.done for t in batch], dtype=torch.float32, device=self.device
        )

        self.online.train()
        q_sa = self.online(sa)

        # Bootstrap max Q over legal next actions (variable count per sample).
        with torch.no_grad():
            next_q_max = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
            for i, t in enumerate(batch):
                if t.done or t.next_state_actions.shape[0] == 0:
                    next_q_max[i] = 0.0
                    continue
                next_x = torch.as_tensor(
                    t.next_state_actions, dtype=torch.float32, device=self.device
                )
                if cfg.double_dqn:
                    # Online network selects action; target evaluates it.
                    q_online = self.online(next_x)
                    best = int(torch.argmax(q_online).item())
                    next_q_max[i] = self.target(next_x[best : best + 1])[0]
                else:
                    next_q_max[i] = self.target(next_x).max()

            targets = rewards + cfg.gamma * next_q_max * (1.0 - dones)

        return F.smooth_l1_loss(q_sa, targets)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
                "grad_steps": self.grad_steps,
                "learn_steps": self.learn_steps,
            },
            path,
        )

    def load(self, path: str | Path, load_optimizer: bool = True) -> None:
        path = Path(path)
        # weights_only=False: checkpoint includes config dict / non-tensor fields.
        ckpt: dict[str, Any] = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        if load_optimizer and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = float(ckpt.get("epsilon", self.config.epsilon_end))
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.grad_steps = int(ckpt.get("grad_steps", 0))
        self.learn_steps = int(ckpt.get("learn_steps", 0))
        self.online.to(self.device)
        self.target.to(self.device)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: Optional[str] = None,
        explore: bool = False,
    ) -> "DQNAgent":
        path = Path(path)
        ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})
        # Filter unknown keys for forward compatibility
        valid = {f.name for f in DQNConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = DQNConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
        if device is not None:
            cfg.device = device
        agent = cls(cfg)
        agent.load(path)
        if not explore:
            agent.epsilon = 0.0
        return agent
