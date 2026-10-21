"""
Persistent expert-imitation regulariser for Double DQN.

    L_total = L_TD + lambda * L_imit

``L_imit`` is the listwise cross-entropy of softmax(Q_online over the *legal*
placements of a demonstration state) against the heuristic's chosen placement.
It is the same loss, state representation and legal-action ordering that were
used for behaviour cloning (``demos.listwise_ce``); demonstration states come
from a persistent ``DemoSet`` that is separate from the replay buffer.

Properties (all covered by tests/test_imitation.py):
  * differentiable through the online Q-network only; labels are fixed integers
    from the stored heuristic choice, the target network is never touched;
  * computed on legal actions only: the network is run on the *valid* rows
    (``feats[mask]``) and scattered into a padded logit matrix whose padding
    slots are -1e9, so padding rows can neither receive probability mass nor
    influence the loss/gradient (and padded zero-rows never reach LayerNorm);
  * the mask is built from the real legal-set length of each state and every
    label is checked to be a legal index.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from tetris_rl.training.demos import DemoSet, _padded_legal

NEG = -1e9


def masked_logits(net: nn.Module, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Q-values (B, A) for padded legal-action features; padded slots = NEG."""
    b, a, f = feats.shape
    if mask.shape != (b, a):
        raise ValueError(f"mask shape {tuple(mask.shape)} != feats (B, A) {(b, a)}")
    q = torch.full((b, a), NEG, dtype=feats.dtype, device=feats.device)
    q[mask] = net(feats[mask])  # only legal rows are evaluated
    return q


def imitation_ce(
    net: nn.Module,
    feats: torch.Tensor,
    mask: torch.Tensor,
    chosen: torch.Tensor,
    *,
    with_stats: bool = True,
) -> tuple[torch.Tensor, dict]:
    """
    Cross-entropy of softmax(Q over legal actions) vs the expert's action.

    feats: (B, A, F) padded legal-action features; mask: (B, A) True = legal;
    chosen: (B,) index into the legal list (must be legal).
    Returns (scalar loss with grad through ``net``, stats dict of floats).
    """
    if chosen.dtype != torch.long:
        chosen = chosen.long()
    if not bool(mask.any(dim=1).all()):
        raise ValueError("a demo state has no legal action")
    if not bool(mask.gather(1, chosen.unsqueeze(1)).all()):
        raise ValueError("expert label points at a padded / illegal slot")
    logits = masked_logits(net, feats, mask)
    loss = torch.nn.functional.cross_entropy(logits, chosen)
    stats: dict = {}
    if with_stats:
        with torch.no_grad():
            lg = logits.detach()
            q_exp = lg.gather(1, chosen.unsqueeze(1)).squeeze(1)
            other = lg.scatter(1, chosen.unsqueeze(1), NEG)
            best_other = other.max(dim=1).values
            n_legal = mask.sum(dim=1).clamp(min=1)
            q_mean = (lg.masked_fill(~mask, 0.0).sum(dim=1)) / n_legal
            stats = {
                "top1": float((lg.argmax(dim=1) == chosen).float().mean().item()),
                "expert_prob": float(torch.softmax(lg, dim=1).gather(1, chosen.unsqueeze(1)).mean().item()),
                # expert Q minus best alternative Q (>0 <=> expert ranked first)
                # (states with a single legal action have no alternative: excluded)
                "margin": float(
                    (q_exp - best_other)[n_legal >= 2].mean().item()
                    if bool((n_legal >= 2).any())
                    else 0.0
                ),
                # expert Q relative to the mean legal Q (the "relative score")
                "rel_score": float((q_exp - q_mean).mean().item()),
            }
    return loss, stats


class ImitationRegularizer:
    """Samples (feats, mask, chosen) batches from a persistent DemoSet."""

    def __init__(
        self,
        demos: DemoSet,
        weight: float,
        batch_size: int = 64,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        if weight < 0:
            raise ValueError("imitation weight must be >= 0")
        self.demos = demos
        self.weight = float(weight)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)  # independent of the agent / buffer RNG

    def sample_indices(self) -> np.ndarray:
        return self.rng.integers(0, len(self.demos), size=self.batch_size)

    def tensors(self, idx: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _padded_legal(self.demos, idx, self.device)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.tensors(self.sample_indices())


@torch.no_grad()
def demo_metrics(
    net: nn.Module, demos: DemoSet, idx: np.ndarray, chunk: int = 250
) -> dict:
    """Mean CE / top-1 / margin / expert prob on a fixed index set (no grad)."""
    ces, tops, margins, probs, rels, ns = [], [], [], [], [], []
    for part in np.array_split(idx, max(1, len(idx) // chunk)):
        feats, mask, chosen = _padded_legal(demos, part, torch.device("cpu"))
        feats, mask, chosen = (t.to(next(net.parameters()).device) for t in (feats, mask, chosen))
        loss, st = imitation_ce(net, feats, mask, chosen)
        ces.append(float(loss.item()))
        tops.append(st["top1"])
        margins.append(st["margin"])
        probs.append(st["expert_prob"])
        rels.append(st["rel_score"])
        ns.append(len(part))
    w = np.asarray(ns, dtype=np.float64)
    w /= w.sum()
    avg = lambda x: float(np.sum(np.asarray(x) * w))  # noqa: E731
    return {
        "ce": avg(ces),
        "top1": avg(tops),
        "margin": avg(margins),
        "expert_prob": avg(probs),
        "rel_score": avg(rels),
    }


def fixed_subset(demos: DemoSet, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(demos), size=min(n, len(demos)), replace=False))


def flat_grad_norm(grads) -> float:
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += float((g.detach().double() ** 2).sum().item())
    return sq**0.5


def grad_cosine(g1, g2) -> float:
    dot = n1 = n2 = 0.0
    for a, b in zip(g1, g2):
        if a is None or b is None:
            continue
        a, b = a.detach().double(), b.detach().double()
        dot += float((a * b).sum().item())
        n1 += float((a * a).sum().item())
        n2 += float((b * b).sum().item())
    return dot / (n1**0.5 * n2**0.5) if n1 > 0 and n2 > 0 else float("nan")


__all__ = [
    "ImitationRegularizer",
    "demo_metrics",
    "fixed_subset",
    "flat_grad_norm",
    "grad_cosine",
    "imitation_ce",
    "masked_logits",
]
