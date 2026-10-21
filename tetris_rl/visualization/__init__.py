"""
Visualisation tools for the placement-action DQN (separate from training/eval).

Modules
-------
model_info            introspect a loaded ``QNetwork`` and the 232-d feature layout
analysis              score every legal placement with the real DQN / heuristic
architecture_diagram  publication-quality SVG + PNG of the network
explorer              local browser app (``python -m tetris_rl.visualization.explorer``)

Nothing here modifies the engine, features, agents or checkpoints; it only
*reads* them.
"""
