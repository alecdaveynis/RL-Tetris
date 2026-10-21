"""
Heuristic demonstrations and behaviour-cloning pretraining.

Demonstrations are produced by the Dellacherie heuristic on seeds disjoint
from training / validation / diagnostic / test seeds. Each decision stores
the full legal-action feature set, so demo transitions are directly usable
as replay-buffer entries (state_action, reward, next legal-action set,
terminated) and as listwise imitation targets.

The heuristic is NEVER used at evaluation time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tetris_rl.environment import TetrisEnv
from tetris_rl.features import placement_feature_batch, state_action_feature_size
from tetris_rl.rewards import RewardConfig
from tetris_rl.training.diagnostics import _best_set, heuristic_scores_from_features
from tetris_rl.training.replay_buffer import Transition

DEMO_TRAIN_SEEDS: tuple[int, ...] = tuple(range(300_000, 300_050))
DEMO_VAL_SEEDS: tuple[int, ...] = tuple(range(310_000, 310_010))


def heuristic_choice(feats: np.ndarray, rng: np.random.Generator) -> int:
    """Heuristic action from a candidate-feature batch (ties broken randomly)."""
    best = _best_set(heuristic_scores_from_features(feats))
    return int(rng.choice(best))


def collect_demonstrations(
    seeds: Sequence[int],
    reward_config: RewardConfig,
    max_pieces: int = 500,
    tie_seed: int = 0,
) -> dict[str, np.ndarray]:
    """Play the heuristic on ``seeds`` and return a ragged-array dict (npz-ready)."""
    rng = np.random.default_rng(tie_seed)
    blocks: list[np.ndarray] = []
    t_state, t_next, t_chosen, t_reward, t_term, t_game = [], [], [], [], [], []
    for gi, seed in enumerate(seeds):
        env = TetrisEnv(seed=int(seed), reward_config=reward_config)
        env.reset(seed=int(seed))
        if not env.legal_placements():
            continue
        cur = placement_feature_batch(env.engine, env.legal_placements())
        cur_block = len(blocks)
        blocks.append(cur)
        while True:
            a = heuristic_choice(cur, rng)
            _, reward, terminated, _, _ = env.step(a)
            truncated = (not terminated) and env.engine.pieces_placed >= max_pieces
            t_state.append(cur_block)
            t_chosen.append(a)
            t_reward.append(float(reward))
            t_term.append(bool(terminated))
            t_game.append(gi)
            if terminated:
                t_next.append(-1)
                break
            nxt = placement_feature_batch(env.engine, env.legal_placements())
            nxt_block = len(blocks)
            blocks.append(nxt)
            t_next.append(nxt_block)
            if truncated:
                break
            cur, cur_block = nxt, nxt_block
    lens = np.array([b.shape[0] for b in blocks], dtype=np.int64)
    return {
        "feats": np.concatenate(blocks, axis=0).astype(np.float32),
        "block_off": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        "t_state": np.array(t_state, dtype=np.int64),
        "t_next": np.array(t_next, dtype=np.int64),
        "t_chosen": np.array(t_chosen, dtype=np.int64),
        "t_reward": np.array(t_reward, dtype=np.float32),
        "t_term": np.array(t_term, dtype=bool),
        "t_game": np.array(t_game, dtype=np.int64),
    }


def save_demos(path: str | Path, demos: dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **demos)


@dataclass
class DemoSet:
    d: dict[str, np.ndarray]

    @classmethod
    def load(cls, path: str | Path) -> "DemoSet":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def __len__(self) -> int:
        return int(self.d["t_state"].shape[0])

    def block(self, b: int) -> np.ndarray:
        off = self.d["block_off"]
        return self.d["feats"][off[b] : off[b + 1]]

    def transition(self, i: int) -> Transition:
        d = self.d
        sa = self.block(int(d["t_state"][i]))[int(d["t_chosen"][i])]
        term = bool(d["t_term"][i])
        nb = int(d["t_next"][i])
        nxt = (
            np.zeros((0, state_action_feature_size()), dtype=np.float32)
            if term or nb < 0
            else self.block(nb)
        )
        return Transition(sa, float(d["t_reward"][i]), nxt, term)

    def fill_buffer(self, buffer) -> int:
        for i in range(len(self)):
            t = self.transition(i)
            buffer.add(t.state_action, t.reward, t.next_state_actions, terminated=t.terminated)
        return len(self)

    def mean_return_stats(self) -> dict:
        d = self.d
        return {
            "n_transitions": len(self),
            "n_games": int(d["t_game"].max()) + 1 if len(self) else 0,
            "mean_reward": float(d["t_reward"].mean()),
            "terminal_frac": float(d["t_term"].mean()),
            "mean_legal_actions": float(np.diff(d["block_off"]).mean()),
        }


def _padded_legal(demos: DemoSet, idx: np.ndarray, device: torch.device):
    blocks = [demos.block(int(demos.d["t_state"][i])) for i in idx]
    max_a = max(b.shape[0] for b in blocks)
    f = demos.d["feats"].shape[1]
    feats = np.zeros((len(idx), max_a, f), dtype=np.float32)
    mask = np.zeros((len(idx), max_a), dtype=bool)
    for k, b in enumerate(blocks):
        feats[k, : b.shape[0]] = b
        mask[k, : b.shape[0]] = True
    chosen = demos.d["t_chosen"][idx]
    return (
        torch.as_tensor(feats, device=device),
        torch.as_tensor(mask, device=device),
        torch.as_tensor(chosen, device=device),
    )


def listwise_ce(agent, demos: DemoSet, idx: np.ndarray) -> tuple[torch.Tensor, float]:
    """Cross-entropy of softmax(Q over legal actions) vs the heuristic's choice; also top-1 acc."""
    feats, mask, chosen = _padded_legal(demos, idx, agent.device)
    b, a, f = feats.shape
    q = agent.online(feats.reshape(b * a, f)).reshape(b, a)
    logits = q.masked_fill(~mask, -1e9)
    loss = F.cross_entropy(logits, chosen)
    acc = float((logits.argmax(dim=1) == chosen).float().mean().item())
    return loss, acc


@torch.no_grad()
def demo_top1(agent, demos: DemoSet, n: int = 2000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(demos), size=min(n, len(demos)), replace=False)
    agent.online.eval()
    accs = []
    for chunk in np.array_split(idx, max(1, len(idx) // 250)):
        feats, mask, chosen = _padded_legal(demos, chunk, agent.device)
        b, a, f = feats.shape
        q = agent.online(feats.reshape(b * a, f)).reshape(b, a).masked_fill(~mask, -1e9)
        accs.append((q.argmax(dim=1) == chosen).float().cpu().numpy())
    return float(np.concatenate(accs).mean())


def bc_pretrain(
    agent,
    demos: DemoSet,
    steps: int,
    batch_size: int = 64,
    td_weight: float = 1.0,
    val_demos: Optional[DemoSet] = None,
    seed: int = 0,
    log_every: int = 500,
) -> list[dict]:
    """
    Pre-train with listwise CE on the heuristic's choice + 1-step TD loss on the
    same demo transitions (keeps Q on the reward scale). Does not advance the
    agent's env-step / epsilon / grad-step counters, and uses its own RNG.
    """
    rng = np.random.default_rng(seed)
    cfg = agent.config
    log: list[dict] = []
    ces, accs, tds = [], [], []
    for step in range(1, steps + 1):
        idx = rng.integers(0, len(demos), size=batch_size)
        agent.online.train()
        ce, acc = listwise_ce(agent, demos, idx)
        loss = ce
        if td_weight > 0:
            batch = [demos.transition(int(i)) for i in idx]
            td_loss = agent._compute_loss(batch)
            loss = ce + td_weight * td_loss
            tds.append(float(td_loss.item()))
        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.online.parameters(), cfg.grad_clip)
        agent.optimizer.step()
        if step % cfg.target_update_freq == 0:
            agent.target.load_state_dict(agent.online.state_dict())
        ces.append(float(ce.item()))
        accs.append(acc)
        if step % log_every == 0 or step == steps:
            rec = {
                "bc_step": step,
                "ce": float(np.mean(ces)),
                "train_top1": float(np.mean(accs)),
                "td": float(np.mean(tds)) if tds else float("nan"),
                "val_top1": demo_top1(agent, val_demos) if val_demos is not None else float("nan"),
            }
            log.append(rec)
            ces, accs, tds = [], [], []
    agent.target.load_state_dict(agent.online.state_dict())
    return log
