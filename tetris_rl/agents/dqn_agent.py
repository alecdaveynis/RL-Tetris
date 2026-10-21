"""
Placement-action Deep Q-Network for Tetris.

Stable-Baselines3's DQN expects a fixed Discrete action space and does not
support action masking. Instead we train a Q-network that scores each legal
placement from its state-action feature vector:

    Q(s, a) = f_θ(φ(s, a))

At decision time we enumerate legal placements, compute φ for each, and pick
argmax_a Q(s, a) (ε-greedy during training). Bellman targets bootstrap from
max over legal next actions (or Double DQN: online argmax, target evaluation).

True terminals (game over) do not bootstrap. Time-limit truncations do.
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

CHECKPOINT_VERSION = 2


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
    """Post-audit defaults (config B). Network/reward weights unchanged here."""
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
        self.best_eval_score = -float("inf")
        self.episodes_trained = 0
        self.last_train_stats: dict[str, float] = {}
        self.imitation: Optional[Any] = None  # optional ImitationRegularizer
        self.grad_diag_every = 50

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
        terminated: bool = False,
        truncated: bool = False,
        *,
        done: Optional[bool] = None,
    ) -> None:
        """
        Store transition after env.step.

        Bellman bootstrap is disabled only when ``terminated`` (true game over).
        Truncations (e.g. max-pieces cap) still store next-action features and
        bootstrap. Legacy ``done=True`` is treated as terminated.
        """
        if done is not None:
            terminated = bool(done)
            truncated = False

        if terminated:
            next_feats = np.zeros((0, self.feature_dim), dtype=np.float32)
        else:
            # Includes truncated episodes: bootstrap from the post-step legal set.
            next_feats = placement_feature_batch(env.engine, env.legal_placements())

        self.buffer.add(
            env_before_features,
            reward,
            next_feats,
            terminated=bool(terminated),
        )
        self.total_steps += 1
        self._decay_epsilon()

    def features_for_action(self, env: TetrisEnv, action: int) -> np.ndarray:
        return state_action_features(env.engine, env.legal_placements()[action])

    def attach_imitation(self, regularizer: Any, grad_diag_every: int = 50) -> None:
        """
        Attach a persistent expert-imitation regulariser (see training/imitation.py).
        Total loss = TD + regularizer.weight * CE(demo batch). With weight 0 the CE
        is only *logged* (no gradient), so the update is identical to plain DQN.
        """
        self.imitation = regularizer
        self.grad_diag_every = int(grad_diag_every)

    def load_init_weights(self, path: str | Path, load_optimizer: bool = True) -> None:
        """
        Initialise from a (behaviour-cloned) checkpoint: copies online / target
        weights (and optionally Adam state) only. Counters, epsilon, replay buffer
        and RNG state are NOT taken from the checkpoint -- this is a fresh RL run.
        """
        ckpt: dict[str, Any] = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        if load_optimizer and ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.lr  # CLI/preset lr wins over the checkpoint's
        self.epsilon = self.config.epsilon_start

    def train_step(self) -> Optional[float]:
        """One gradient update. Returns the total loss, or None if buffer too small."""
        cfg = self.config
        if not self.buffer.can_sample(cfg.batch_size):
            return None

        batch = self.buffer.sample(cfg.batch_size)
        td_loss = self._compute_loss(batch)
        loss = td_loss
        reg = getattr(self, "imitation", None)
        if reg is not None:
            loss = self._add_imitation(td_loss, reg)
        else:
            self.last_train_stats["td_loss"] = float(td_loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        # clip_grad_norm_ returns the pre-clip total norm (instrumentation only).
        grad_norm = nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.last_train_stats["grad_norm"] = float(grad_norm)
        self.optimizer.step()

        self.grad_steps += 1
        self.learn_steps += 1
        if self.grad_steps % cfg.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    def _add_imitation(self, td_loss: torch.Tensor, reg: Any) -> torch.Tensor:
        """
        L_total = td_loss + weight * CE(Q_online over legal demo actions, expert action).
        Gradient flows only through ``self.online``; expert labels are stored integers
        and the target network is not used by the CE term.
        """
        from tetris_rl.training.imitation import flat_grad_norm, grad_cosine, imitation_ce

        feats, mask, chosen = reg.sample()
        diag = self.grad_diag_every > 0 and (self.grad_steps % self.grad_diag_every == 0)
        w = reg.weight
        st = self.last_train_stats
        if w > 0 or diag:
            ce, ce_stats = imitation_ce(self.online, feats, mask, chosen)
        else:  # lambda = 0: log-only, no graph
            with torch.no_grad():
                ce, ce_stats = imitation_ce(self.online, feats, mask, chosen)
        total = td_loss + w * ce if w > 0 else td_loss
        st.update(
            td_loss=float(td_loss.item()),
            imit_loss=float(ce.item()),
            imit_weighted=float(w * ce.item()),
            total_loss=float(td_loss.item() + w * ce.item()),
            imit_top1=ce_stats["top1"],
            imit_margin=ce_stats["margin"],
            imit_expert_prob=ce_stats["expert_prob"],
        )
        if diag:
            # Instrumentation only: per-loss gradient norms (no .grad is written).
            params = [p for p in self.online.parameters() if p.requires_grad]
            g_td = torch.autograd.grad(td_loss, params, retain_graph=True, allow_unused=True)
            g_im = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
            st.update(
                gn_td=flat_grad_norm(g_td),
                gn_imit_unweighted=flat_grad_norm(g_im),
                gn_imit_weighted=w * flat_grad_norm(g_im),
                grad_cos_td_imit=grad_cosine(g_td, g_im),
            )
        return total

    def _pad_next_actions(
        self, batch: list[Transition]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stack next-action features to (B, A_max, F) with a boolean mask."""
        bsz = len(batch)
        lengths = [0 if t.terminated else int(t.next_state_actions.shape[0]) for t in batch]
        max_a = max(lengths) if lengths else 0
        if max_a == 0:
            feats = torch.zeros(bsz, 0, self.feature_dim, device=self.device)
            mask = torch.zeros(bsz, 0, dtype=torch.bool, device=self.device)
            return feats, mask

        feats = torch.zeros(bsz, max_a, self.feature_dim, device=self.device)
        mask = torch.zeros(bsz, max_a, dtype=torch.bool, device=self.device)
        for i, t in enumerate(batch):
            n = lengths[i]
            if n == 0:
                continue
            feats[i, :n] = torch.as_tensor(
                t.next_state_actions, dtype=torch.float32, device=self.device
            )
            mask[i, :n] = True
        return feats, mask

    def _next_q_max_vectorized(
        self, next_feats: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bootstrap values for a padded next-action batch.

        next_feats: (B, A, F), mask: (B, A). Returns (B,).
        """
        bsz, max_a, feat_dim = next_feats.shape
        if max_a == 0:
            return torch.zeros(bsz, dtype=torch.float32, device=self.device)

        flat = next_feats.reshape(bsz * max_a, feat_dim)
        cfg = self.config

        if cfg.double_dqn:
            q_online = self.online(flat).reshape(bsz, max_a)
            q_target = self.target(flat).reshape(bsz, max_a)
            # Mask illegal / pad slots before argmax.
            q_online_masked = q_online.masked_fill(~mask, -1e9)
            best = q_online_masked.argmax(dim=1)
            gathered = q_target.gather(1, best.unsqueeze(1)).squeeze(1)
            has_any = mask.any(dim=1)
            return torch.where(has_any, gathered, torch.zeros_like(gathered))

        q_target = self.target(flat).reshape(bsz, max_a)
        q_masked = q_target.masked_fill(~mask, -1e9)
        vals, _ = q_masked.max(dim=1)
        has_any = mask.any(dim=1)
        return torch.where(has_any, vals, torch.zeros_like(vals))

    def _next_q_max_loop(self, batch: list[Transition]) -> torch.Tensor:
        """Reference per-sample loop (for tests / benchmarks)."""
        cfg = self.config
        next_q_max = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
        for i, t in enumerate(batch):
            if t.terminated or t.next_state_actions.shape[0] == 0:
                next_q_max[i] = 0.0
                continue
            next_x = torch.as_tensor(
                t.next_state_actions, dtype=torch.float32, device=self.device
            )
            if cfg.double_dqn:
                q_online = self.online(next_x)
                best = int(torch.argmax(q_online).item())
                next_q_max[i] = self.target(next_x[best : best + 1])[0]
            else:
                next_q_max[i] = self.target(next_x).max()
        return next_q_max

    def bellman_targets(
        self, batch: list[Transition], *, vectorized: bool = True
    ) -> torch.Tensor:
        """Compute Bellman targets for a batch (no grad). Useful for tests."""
        cfg = self.config
        rewards = torch.as_tensor(
            [t.reward for t in batch], dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(
            [t.terminated for t in batch], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            if vectorized:
                next_feats, mask = self._pad_next_actions(batch)
                next_q_max = self._next_q_max_vectorized(next_feats, mask)
            else:
                next_q_max = self._next_q_max_loop(batch)
            return rewards + cfg.gamma * next_q_max * (1.0 - terminated)

    def _compute_loss(self, batch: list[Transition]) -> torch.Tensor:
        sa = torch.as_tensor(
            np.stack([t.state_action for t in batch], axis=0),
            dtype=torch.float32,
            device=self.device,
        )
        self.online.train()
        q_sa = self.online(sa)
        with torch.no_grad():
            targets = self.bellman_targets(batch, vectorized=True)
            td = q_sa.detach() - targets
            self.last_train_stats = {
                "q_mean": float(q_sa.detach().mean().item()),
                "q_abs_mean": float(q_sa.detach().abs().mean().item()),
                "td_error": float(td.abs().mean().item()),
                "target_mean": float(targets.mean().item()),
            }
        return F.smooth_l1_loss(q_sa, targets)

    def save(self, path: str | Path, include_buffer: bool = True) -> None:
        """Save checkpoint. ``include_buffer=False`` writes a small weights/optimizer file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.config),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "grad_steps": self.grad_steps,
            "learn_steps": self.learn_steps,
            "best_eval_score": self.best_eval_score,
            "episodes_trained": self.episodes_trained,
            "buffer": self.buffer.state_dict() if include_buffer else None,
            "agent_rng_state": self.rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
        }
        if self.device.type == "cuda":
            payload["cuda_rng_state"] = torch.cuda.get_rng_state()
        torch.save(payload, path)

    def load(self, path: str | Path, load_optimizer: bool = True) -> None:
        path = Path(path)
        ckpt: dict[str, Any] = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        if load_optimizer and "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception:
                # Architecture/hparams mismatch on older checkpoints — keep fresh opt.
                pass
        self.epsilon = float(ckpt.get("epsilon", self.config.epsilon_end))
        self.total_steps = int(ckpt.get("total_steps", 0))
        self.grad_steps = int(ckpt.get("grad_steps", 0))
        self.learn_steps = int(ckpt.get("learn_steps", 0))
        self.best_eval_score = float(ckpt.get("best_eval_score", -float("inf")))
        self.episodes_trained = int(ckpt.get("episodes_trained", 0))

        if "buffer" in ckpt and ckpt["buffer"] is not None:
            self.buffer.load_state_dict(ckpt["buffer"])

        if "agent_rng_state" in ckpt and ckpt["agent_rng_state"] is not None:
            try:
                self.rng.bit_generator.state = ckpt["agent_rng_state"]
            except Exception:
                pass

        # torch.load(map_location=mps/cuda) can move the CPU ByteTensor RNG
        # state onto the accelerator; set_rng_state requires a CPU ByteTensor.
        self._restore_torch_rng(ckpt.get("torch_rng_state"))
        if "numpy_rng_state" in ckpt and ckpt["numpy_rng_state"] is not None:
            try:
                np.random.set_state(ckpt["numpy_rng_state"])
            except Exception:
                pass
        if (
            self.device.type == "cuda"
            and "cuda_rng_state" in ckpt
            and ckpt["cuda_rng_state"] is not None
        ):
            try:
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
            except Exception:
                pass

        self.online.to(self.device)
        self.target.to(self.device)

    @staticmethod
    def _restore_torch_rng(state: Any) -> None:
        """Restore torch CPU RNG; ignore unusable states (e.g. MPS-mapped tensors)."""
        if state is None:
            return
        try:
            if isinstance(state, torch.Tensor):
                state = state.detach().to("cpu").contiguous()
                if state.dtype != torch.uint8:
                    state = state.to(torch.uint8)
            torch.set_rng_state(state)
        except (TypeError, RuntimeError):
            return

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
        valid = {f.name for f in DQNConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        cfg = DQNConfig(**{k: v for k, v in cfg_dict.items() if k in valid})
        if device is not None:
            cfg.device = device
        # Avoid re-seeding from config.seed before load restores RNG states.
        seed = cfg.seed
        cfg.seed = None
        agent = cls(cfg)
        cfg.seed = seed
        agent.config.seed = seed
        agent.load(path)
        if not explore:
            agent.epsilon = 0.0
        return agent
