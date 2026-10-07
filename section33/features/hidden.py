"""Fixed random projections of hidden states (H baseline storage).

Plan section 14: store scalar summaries and fixed random projections broadly;
store exact raw vectors only for the training-fitted PCA path and an audit
subset, or replay tokens later.  This module implements the fixed projection
part.  The projection matrices are seeded and must never be fitted on data.

The layer set is frozen as *relative depths* so that the same positions are
used for every model (plan section 5.1: "Map the fixed layer set by relative
depth").  For a 28-layer model the fractions reproduce the original absolute
set ``(0, 4, 8, 12, 16, 20, 24, 27)``.
"""

from __future__ import annotations

import numpy as np
import torch

#: Absolute reference set used in the pilot (Qwen2.5-7B, 28 layers).
DEFAULT_PROJ_LAYERS: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 27)
_REFERENCE_DEPTH = 28
#: Frozen relative depths derived from the reference set.
PROJ_LAYER_FRACTIONS: tuple[float, ...] = tuple(
    layer / (_REFERENCE_DEPTH - 1) for layer in DEFAULT_PROJ_LAYERS
)


def proj_layers(n_layers: int) -> tuple[int, ...]:
    """Map the frozen relative depths onto a model with ``n_layers`` layers."""

    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if n_layers < 2:
        return (0,)
    layers = {round(fraction * (n_layers - 1)) for fraction in PROJ_LAYER_FRACTIONS}
    return tuple(sorted(layer for layer in layers if 0 <= layer < n_layers))


class HiddenProjector:
    """Seeded random projections of pre-block and post-block hidden states."""

    def __init__(self, n_layers: int, d_model: int, dim: int = 64, seed: int = 11):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.n_layers = n_layers
        self.layers = proj_layers(n_layers)
        self.pre = [
            torch.tensor(rng.normal(size=(dim, d_model)), dtype=torch.float32)
            for _ in range(n_layers)
        ]
        self.post = [
            torch.tensor(rng.normal(size=(dim, d_model)), dtype=torch.float32)
            for _ in range(n_layers)
        ]

    def project(self, hidden: torch.Tensor, layer: int, kind: str) -> torch.Tensor:
        matrix = (self.pre if kind == "pre" else self.post)[layer].to(hidden.device)
        return torch.einsum("...d,kd->...k", hidden.float(), matrix)

    @staticmethod
    def names(kind: str, position: str, layer: int, dim: int) -> list[str]:
        return [f"hiddenproj__{kind}__{position}__l{layer}__k{index}" for index in range(dim)]

    def scalar_names(self, position: str) -> list[str]:
        names: list[str] = []
        for layer in self.layers:
            names.extend(self.names("pre", position, layer, self.dim))
            names.extend(self.names("post", position, layer, self.dim))
        return names
