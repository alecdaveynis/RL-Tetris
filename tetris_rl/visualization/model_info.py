"""
Introspection helpers: what does the trained Q-network actually look like?

Everything is derived from the *instantiated* ``QNetwork`` (modules and weight
shapes) and from the constants in ``tetris_rl.engine`` / ``tetris_rl.features``,
so the diagrams cannot drift from the implementation.

The feature *layout* below mirrors ``features.state_action_features``; the unit
tests in ``tests/test_visualization.py`` check it against the real function
(e.g. that the "holes" slot really contains ``holes / (H*W)``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from tetris_rl.agents.dqn_agent import QNetwork
from tetris_rl.engine import BOARD_HEIGHT, BOARD_WIDTH, PIECE_NAMES
from tetris_rl.features import state_action_feature_size

# --------------------------------------------------------------------------
# Input feature layout  φ(s, a)  ∈ R^232
# --------------------------------------------------------------------------

#: Names of the 8 scalar slots at the end of φ, in order (see features.py).
SCALAR_FEATURES: tuple[tuple[str, str], ...] = (
    ("aggregate_height", "aggregate height / (H·W)"),
    ("holes", "holes / (H·W)"),
    ("bumpiness", "bumpiness / (H·W)"),
    ("max_height", "max height / H"),
    ("lines_cleared", "lines cleared / 4"),
    ("delta_holes", "Δ holes / (H·W)"),
    ("delta_aggregate_height", "Δ aggregate height / (H·W)"),
    ("delta_bumpiness", "Δ bumpiness / (H·W)"),
)


@dataclass(frozen=True)
class FeatureGroup:
    key: str
    label: str
    start: int
    size: int
    detail: str

    @property
    def stop(self) -> int:
        return self.start + self.size


def feature_groups(width: int = BOARD_WIDTH, height: int = BOARD_HEIGHT) -> list[FeatureGroup]:
    """Contiguous slices of the state-action feature vector, in order."""
    spec = [
        ("board", "Resulting board", height * width,
         f"{height}×{width} occupancy after the piece locks and full rows clear (0/1)"),
        ("current", "Current piece", 7, "one-hot over I O T S Z J L"),
        ("next", "Next piece", 7, "one-hot preview piece"),
        ("heights", "Column heights", width, "height of each column / board height"),
        ("scalars", "Board metrics", len(SCALAR_FEATURES),
         "holes, height, bumpiness, lines cleared + before→after deltas"),
    ]
    groups: list[FeatureGroup] = []
    pos = 0
    for key, label, size, detail in spec:
        groups.append(FeatureGroup(key, label, pos, size, detail))
        pos += size
    return groups


def feature_names(width: int = BOARD_WIDTH, height: int = BOARD_HEIGHT) -> list[str]:
    """Human-readable name for each of the 232 input slots."""
    names: list[str] = []
    for r in range(height):
        for c in range(width):
            names.append(f"board[row {r}, col {c}]")
    names += [f"current piece = {p}" for p in PIECE_NAMES]
    names += [f"next piece = {p}" for p in PIECE_NAMES]
    names += [f"height of column {c} / {height}" for c in range(width)]
    names += [label for _, label in SCALAR_FEATURES]
    assert len(names) == state_action_feature_size(width, height)
    return names


# --------------------------------------------------------------------------
# Network description
# --------------------------------------------------------------------------


@dataclass
class LayerInfo:
    """One computational stage of the Q-network."""

    kind: str  # "input" | "layernorm" | "linear" | "output"
    name: str
    in_dim: int
    out_dim: int
    activation: Optional[str] = None  # "ReLU" for hidden Linear layers
    params: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class NetworkInfo:
    input_dim: int
    use_layer_norm: bool
    hidden_sizes: tuple[int, ...]
    layers: list[LayerInfo]
    total_params: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dim": self.input_dim,
            "use_layer_norm": self.use_layer_norm,
            "hidden_sizes": list(self.hidden_sizes),
            "layers": [l.to_dict() for l in self.layers],
            "total_params": self.total_params,
        }


def _n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def describe_network(net: QNetwork) -> NetworkInfo:
    """Read the architecture off an instantiated ``QNetwork`` (not from a config)."""
    linears = [m for m in net.net if isinstance(m, nn.Linear)]
    if not linears:
        raise ValueError("QNetwork has no Linear layers")
    input_dim = linears[0].in_features
    use_ln = isinstance(net.input_norm, nn.LayerNorm)

    layers = [LayerInfo("input", "Input φ(s, a)", input_dim, input_dim)]
    if use_ln:
        layers.append(
            LayerInfo("layernorm", "LayerNorm", input_dim, input_dim,
                      params=_n_params(net.input_norm))
        )
    modules = list(net.net)
    hidden: list[int] = []
    li = 0
    for i, m in enumerate(modules):
        if not isinstance(m, nn.Linear):
            continue
        li += 1
        nxt = modules[i + 1] if i + 1 < len(modules) else None
        act = type(nxt).__name__ if isinstance(nxt, nn.Module) and not isinstance(nxt, nn.Linear) else None
        is_last = m is linears[-1]
        if not is_last:
            hidden.append(m.out_features)
        layers.append(
            LayerInfo(
                "linear",
                f"Linear {li}" if not is_last else "Linear (output)",
                m.in_features,
                m.out_features,
                activation=act,
                params=_n_params(m),
            )
        )
    layers.append(LayerInfo("output", "Q(s, a)", linears[-1].out_features, linears[-1].out_features))
    return NetworkInfo(
        input_dim=input_dim,
        use_layer_norm=use_ln,
        hidden_sizes=tuple(hidden),
        layers=layers,
        total_params=_n_params(net),
    )


# --------------------------------------------------------------------------
# Layer-by-layer forward pass (for activation inspection)
# --------------------------------------------------------------------------


@torch.no_grad()
def forward_with_activations(net: QNetwork, x: np.ndarray) -> list[dict[str, Any]]:
    """
    Run ``QNetwork.forward`` one module at a time and record every intermediate.

    Returns a list of dicts ``{"name", "kind", "values": (N, dim) ndarray}`` in
    execution order: input, [layernorm], then one entry per Linear (pre-activation,
    "linear") followed by its ReLU ("relu"). The last entry is Q (shape (N, 1)).
    Equivalent to ``net(x)``; the unit tests assert that.
    """
    net.eval()
    h = torch.as_tensor(x, dtype=torch.float32)
    out: list[dict[str, Any]] = [{"name": "input", "kind": "input", "values": h.numpy().copy()}]
    if isinstance(net.input_norm, nn.LayerNorm):
        h = net.input_norm(h)
        out.append({"name": "layernorm", "kind": "layernorm", "values": h.numpy().copy()})
    n_lin = 0
    for m in net.net:
        h = m(h)
        if isinstance(m, nn.Linear):
            n_lin += 1
            out.append({"name": f"linear{n_lin}", "kind": "linear", "values": h.numpy().copy()})
        else:
            out.append({"name": f"{type(m).__name__.lower()}{n_lin}", "kind": "act",
                        "values": h.numpy().copy()})
    return out


def linear_layers(net: QNetwork) -> list[nn.Linear]:
    return [m for m in net.net if isinstance(m, nn.Linear)]
