"""MLP mathematics for the Section 3.3 feature families.

The installed Qwen2 and Mistral MLP modules use no bias in the three
projections.  For the normalized MLP input ``x``:

    a = W_gate x
    b = W_up x
    s = SiLU(a) * b              # the actual channel selection coefficient
    o = W_down s                 # the MLP output (write)

The contribution of channel ``j`` to the output is ``c_j = s_j * v_j`` where
``v_j`` is column ``j`` of ``W_down``.  Its magnitude is
``e_j = |s_j| * ||v_j||``.  ``x . v_j`` is a literal similarity; it is **not**
the coefficient that selects channel ``j``.  The two quantities get separate
feature families and separate results.

All functions accept arbitrary leading batch dimensions and operate on the last
dimension so that a single token vector and a batch behave the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

EPS = 1e-12


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def mlp_activations(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(a, b, s, o)`` for the gated MLP.  Biases are not used."""

    a = F.linear(x, w_gate)
    b = F.linear(x, w_up)
    s = F.silu(a) * b
    o = F.linear(s, w_down)
    return a, b, s, o


def channel_value_norms(w_down: torch.Tensor) -> torch.Tensor:
    """``||v_j||`` for every column of ``W_down`` (shape ``(d_ff,)``)."""

    return w_down.norm(dim=0)


def contribution_magnitudes(s: torch.Tensor, v_norms: torch.Tensor) -> torch.Tensor:
    """``e_j = |s_j| * ||v_j||`` — the mass of channel ``j`` in the output."""

    return s.abs() * v_norms


# ---------------------------------------------------------------------------
# Concentration summaries on a nonnegative mass vector
# ---------------------------------------------------------------------------
def _mass_to_probability(mass: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    total = mass.sum(dim=-1, keepdim=True)
    zero = total.squeeze(-1) <= EPS
    probability = mass / total.clamp_min(EPS)
    return probability, zero


def normalized_entropy(mass: torch.Tensor) -> torch.Tensor:
    """``-sum(p log p) / log(m)`` on the last dimension.  Zero mass -> 0."""

    probability, zero = _mass_to_probability(mass)
    m = mass.shape[-1]
    ent = -(probability * (probability + EPS).log()).sum(dim=-1)
    ent = ent / math.log(m)
    return torch.where(zero, torch.zeros_like(ent), ent)


def top_k_mass(mass: torch.Tensor, k: int) -> torch.Tensor:
    probability, zero = _mass_to_probability(mass)
    k = min(k, probability.shape[-1])
    values = probability.topk(k, dim=-1).values.sum(dim=-1)
    return torch.where(zero, torch.zeros_like(values), values)


def top1_minus_top2_mass(mass: torch.Tensor) -> torch.Tensor:
    probability, zero = _mass_to_probability(mass)
    k = min(2, probability.shape[-1])
    values = probability.topk(k, dim=-1).values
    if values.shape[-1] == 1:
        diff = values[..., 0]
    else:
        diff = values[..., 0] - values[..., 1]
    return torch.where(zero, torch.zeros_like(diff), diff)


def effective_channels(mass: torch.Tensor) -> torch.Tensor:
    """``1 / sum(p^2)`` — inverse participation ratio.  Zero mass -> 1."""

    probability, zero = _mass_to_probability(mass)
    value = 1.0 / (probability.pow(2).sum(dim=-1).clamp_min(EPS))
    return torch.where(zero, torch.ones_like(value), value)


def concentration_summary(mass: torch.Tensor) -> dict[str, torch.Tensor]:
    """The frozen concentration summaries (plan section 3.2).

    Documented fixed values for zero total mass: each statistic is zero and the
    ``zero_mass`` indicator is one (``effective_channels`` is set to one so that
    a later log is safe, and the zero indicator marks the row).
    """

    return {
        "norm_entropy": normalized_entropy(mass),
        "top1_mass": top_k_mass(mass, 1),
        "top5_mass": top_k_mass(mass, 5),
        "top1_minus_top2": top1_minus_top2_mass(mass),
        "effective_channels": effective_channels(mass),
        "zero_mass": _mass_to_probability(mass)[1].to(mass.dtype),
    }


def signed_summary(v: torch.Tensor) -> dict[str, torch.Tensor]:
    """Positive and negative parts plus the signed total (no softmax)."""

    pos = v.clamp_min(0)
    neg = (-v).clamp_min(0)
    return {
        "pos_total": pos.sum(dim=-1),
        "neg_total": neg.sum(dim=-1),
        "signed_total": v.sum(dim=-1),
        "pos_count": (v > 0).sum(dim=-1).to(v.dtype),
        "neg_count": (v < 0).sum(dim=-1).to(v.dtype),
    }


# ---------------------------------------------------------------------------
# Historical reference-free summaries from the pilot
# ---------------------------------------------------------------------------
def softmax_entropy(v: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(v, dim=-1)
    return -(probability * (probability + EPS).log()).sum(dim=-1)


def entropy_of_squares(v: torch.Tensor) -> torch.Tensor:
    """Entropy of the distribution proportional to ``v**2``."""

    squares = v * v
    logits = (squares + EPS).log()
    return softmax_entropy(logits)


def gini(v: torch.Tensor) -> torch.Tensor:
    values = v.abs().sort(dim=-1).values
    n = values.shape[-1]
    cumulative = values.cumsum(dim=-1)
    denominator = cumulative[..., -1:].clamp_min(EPS)
    return (n + 1 - 2 * (cumulative / denominator).sum(dim=-1)) / n


def kurtosis(v: torch.Tensor) -> torch.Tensor:
    mean = v.mean(dim=-1, keepdim=True)
    std = v.std(dim=-1, keepdim=True).clamp_min(EPS)
    return (((v - mean) / std) ** 4).mean(dim=-1)


def gini_centered(v: torch.Tensor) -> torch.Tensor:
    """Historical gini helper kept for exact replication of ``gini_gate``."""

    return gini(v)


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
@dataclass
class MLPTensors:
    """All tensors needed by the frozen feature families for one position."""

    x: torch.Tensor  # normalized MLP input (module input)
    h_pre: torch.Tensor  # residual-stream input to the block
    h_post: torch.Tensor  # residual-stream output of the block
    a: torch.Tensor
    b: torch.Tensor
    s: torch.Tensor
    o: torch.Tensor
    v_norms: torch.Tensor
    w2_dot: torch.Tensor | None = None  # literal x . v_j scores


def direction_summary(tensors: MLPTensors) -> dict[str, torch.Tensor]:
    """Norm, relative norm, cosine, and cancellation for the MLP output."""

    x, o = tensors.x, tensors.o
    e = contribution_magnitudes(tensors.s, tensors.v_norms)
    x_norm = x.norm(dim=-1)
    o_norm = o.norm(dim=-1)
    e_total = e.sum(dim=-1)
    cos = (x * o).sum(dim=-1) / (x_norm * o_norm + EPS)
    return {
        "out_norm": o_norm,
        "out_to_in_norm": o_norm / (x_norm + EPS),
        "cos_out_in": cos,
        "cancellation_ratio": o_norm / (e_total + EPS),
    }


def to_float_dict(d: dict[str, torch.Tensor]) -> dict[str, list[float]]:
    """Move a dictionary of tensors to Python floats (one value per row)."""

    return {k: v.detach().float().cpu().tolist() for k, v in d.items()}
