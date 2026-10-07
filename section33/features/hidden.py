"""Fixed random projections of hidden states (H baseline storage).

Plan section 14: store scalar summaries and fixed random projections broadly;
store exact raw vectors only for the training-fitted PCA path and an audit
subset, or replay tokens later.  This module implements the fixed projection
part.  The projection matrices are seeded and must never be fitted on data.
"""

from __future__ import annotations

import numpy as np
import torch

#: Evenly spaced layer set for projection storage (frozen in the protocol).
DEFAULT_PROJ_LAYERS: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 27)


class HiddenProjector:
    """Seeded random projections of pre-block and post-block hidden states."""

    def __init__(self, n_layers: int, d_model: int, dim: int = 64, seed: int = 11):
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.layers = tuple(layer for layer in DEFAULT_PROJ_LAYERS if layer < n_layers)
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
