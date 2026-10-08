"""Controlled interventions on MLP channel contributions (plan section 8.3).

The functions operate on the selection coefficient ``s`` and the frozen
``W_down``.  They return a new ``s`` (and the matched random controls) so a
forward hook can replace the module output during generation.

All comparisons use the predeclared caliper (10% on mass and output
perturbation norm).  If no matched random control exists, the caller must
report the comparison as unidentifiable and must not silently use a larger
perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import PROTOCOL
from ..features import mlp_math as mm


@dataclass
class InterventionResult:
    name: str
    s_new: torch.Tensor
    o_new: torch.Tensor
    removed_mass: float
    perturbation_norm: float
    matched: bool
    note: str = ""


def _output(s: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...f,df->...d", s, w_down)


def _mass(s: torch.Tensor, v_norms: torch.Tensor) -> torch.Tensor:
    return mm.contribution_magnitudes(s, v_norms)


def identity(s: torch.Tensor, w_down: torch.Tensor, **_kwargs) -> InterventionResult:
    o = _output(s, w_down)
    return InterventionResult("identity", s.clone(), o, 0.0, 0.0, True, "exact rewrite")


def attenuate_topk(
    s: torch.Tensor,
    w_down: torch.Tensor,
    v_norms: torch.Tensor,
    k: int = 1,
    attenuation: float = 0.0,
) -> InterventionResult:
    """Scale the ``k`` largest contribution channels by ``attenuation``."""

    e = _mass(s, v_norms)
    topk = torch.topk(e, k=min(k, e.shape[-1])).indices
    s_new = s.clone()
    s_new[..., topk] = s_new[..., topk] * attenuation
    o_new = _output(s_new, w_down)
    o_old = _output(s, w_down)
    removed = float((e[..., topk] * (1.0 - attenuation)).sum().item())
    perturbation = float((o_new - o_old).norm().item())
    return InterventionResult(
        f"attenuate_top{k}_x{attenuation}", s_new, o_new, removed, perturbation, True
    )


def random_matched(
    s: torch.Tensor,
    w_down: torch.Tensor,
    v_norms: torch.Tensor,
    k: int,
    target_mass: float,
    target_norm: float,
    caliper: float = PROTOCOL.intervention_caliper,
    max_tries: int = 500,
    seed: int = 0,
) -> InterventionResult:
    """Remove ``k`` random channels with matched mass and perturbation norm."""

    generator = torch.Generator(device=s.device).manual_seed(seed)
    e = _mass(s, v_norms)
    n_channels = s.shape[-1]
    o_old = _output(s, w_down)
    best = None
    for _ in range(max_tries):
        candidate = torch.randperm(n_channels, generator=generator, device=s.device)[:k]
        s_new = s.clone()
        s_new[..., candidate] = 0.0
        o_new = _output(s_new, w_down)
        removed = float(e[..., candidate].sum().item())
        perturbation = float((o_new - o_old).norm().item())
        mass_error = abs(removed - target_mass) / max(target_mass, 1e-12)
        norm_error = abs(perturbation - target_norm) / max(target_norm, 1e-12)
        if mass_error <= caliper and norm_error <= caliper:
            return InterventionResult(
                f"random_matched_k{k}", s_new, o_new, removed, perturbation, True
            )
        score = mass_error + norm_error
        if best is None or score < best[0]:
            best = (score, candidate, s_new, o_new, removed, perturbation)
    _, candidate, s_new, o_new, removed, perturbation = best
    return InterventionResult(
        f"random_unmatched_k{k}",
        s_new,
        o_new,
        removed,
        perturbation,
        False,
        "no random control inside the caliper; comparison unidentifiable",
    )


def flatten_sharpen(
    s: torch.Tensor,
    w_down: torch.Tensor,
    exponent: float,
    restore_norm: bool = False,
) -> InterventionResult:
    """Redistribute contribution mass with a power law, keeping signs.

    The transformation acts on the channel contribution mass
    ``e_j = |s_j| * ||W_down[:, j]||``, which is invariant under the
    function-preserving rescaling ``W_up[:, j] *= k``, ``W_down[:, j] /= k``.
    ``exponent < 1`` flattens (more equal contributions).  ``exponent > 1``
    sharpens (a few channels dominate).  Total mass is preserved before the
    optional norm restoration step.
    """

    v_norms = w_down.norm(dim=0)
    mass = _mass(s, v_norms)
    powered = mass.pow(exponent)
    target = mass.sum(dim=-1, keepdim=True)
    scale = powered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    mass_new = powered * target / scale
    s_new = torch.sign(s) * mass_new / v_norms.clamp_min(1e-12)
    o_old = _output(s, w_down)
    o_new = _output(s_new, w_down)
    if restore_norm:
        ratio = o_old.norm(dim=-1, keepdim=True) / o_new.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        s_new = s_new * ratio
        o_new = _output(s_new, w_down)
    return InterventionResult(
        f"power_{exponent}_restore{int(restore_norm)}",
        s_new,
        o_new,
        float((mass - mass_new).abs().sum().item()),
        float((o_new - o_old).norm().item()),
        True,
        "mass redistribution on |s_j| * ||W_down[:, j]||",
    )


def top_channels(s: torch.Tensor, v_norms: torch.Tensor, k: int = 5) -> list[int]:
    e = _mass(s, v_norms)
    return torch.topk(e, k=min(k, e.shape[-1])).indices.tolist()


# ---------------------------------------------------------------------------
# Negative controls for the audit (plan section 8.2)
# ---------------------------------------------------------------------------
def permuted_labels(labels: np.ndarray, subjects: np.ndarray, seed: int = 0) -> np.ndarray:
    """Permute labels between subjects; labels inside a subject stay together."""

    labels = np.asarray(labels)
    subjects = np.asarray(subjects)
    unique = np.unique(subjects)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    mapping = {subject: unique[order[index]] for index, subject in enumerate(unique)}
    by_subject = {subject: labels[subjects == subject] for subject in unique}
    out = np.empty_like(labels)
    for subject in unique:
        out[subjects == subject] = by_subject[mapping[subject]]
    return out


def shuffle_within_strata(values: np.ndarray, strata: np.ndarray, seed: int = 0) -> np.ndarray:
    """Shuffle rows between subjects inside each stratum (wrong-question control)."""

    values = np.asarray(values)
    strata = np.asarray(strata)
    rng = np.random.default_rng(seed)
    out = values.copy()
    for stratum in np.unique(strata):
        index = np.where(strata == stratum)[0]
        out[index] = values[rng.permutation(index)]
    return out
