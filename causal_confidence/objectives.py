"""Pointwise/pair losses, exact grid expectation, and the sampled reward estimator.

Definitions (plan section 8).  For state ``z`` with empirical success
``p̂_z`` and a deterministic report ``c_z``:

    L_point = 0.5[(c_a - p̂_a)^2 + (c_b - p̂_b)^2]
    L_pair  = [(c_a - c_b) - (p̂_a - p̂_b)]^2
    L       = L_point + λ L_pair

The native reporter emits a distribution ``ρ_φ`` over the 21 grid values.  The
exact expected loss over that distribution uses the first two moments per
state: with ``μ_z = E[C_z]`` and ``V_z = Var(C_z)``,

    E[L_point] = 0.5[(μ_a - p̂_a)^2 + (μ_b - p̂_b)^2 + V_a + V_b]
    E[L_pair]  = [(μ_a - μ_b) - (p̂_a - p̂_b)]^2 + V_a + V_b

Omitting the variance terms changes the objective to squared error of the
mean, which is not the expected reward for an emitted sampled report.

The state-mixture weights and the pair-type weights are applied in separate
weighted sums so that states repeated in several pairs are not accidentally
overweighted (plan section 8).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .protocol import CONFIDENCE_GRID, CONDITION_WEIGHTS, PAIR_TYPES, PROTOCOL

EPS = 1e-12


def grid_tensor(device=None, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(CONFIDENCE_GRID, device=device, dtype=dtype)


def grid_moments(probs: torch.Tensor, grid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(mean, variance)`` of a distribution over the report grid.

    ``probs`` has shape ``(..., n_grid)``; normalization is enforced (a
    tolerance check flags silent probability bugs).
    """

    if grid is None:
        grid = grid_tensor(probs.device, probs.dtype)
    total = probs.sum(dim=-1, keepdim=True)
    if not torch.allclose(total, torch.ones_like(total), atol=1e-4):
        raise ValueError("report distribution does not sum to one")
    mean = (probs * grid).sum(dim=-1)
    second = (probs * grid.pow(2)).sum(dim=-1)
    variance = (second - mean.pow(2)).clamp_min(0.0)
    return mean, variance


def expected_point_loss(
    mu: torch.Tensor,
    var: torch.Tensor,
    target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expected pointwise squared error for a sampled report."""

    loss = (mu - target).pow(2) + var
    if weight is not None:
        return (loss * weight).sum()
    return loss.mean()


def expected_pair_loss(
    mu_a: torch.Tensor,
    var_a: torch.Tensor,
    mu_b: torch.Tensor,
    var_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expected contrast loss for two sampled reports of the same question."""

    contrast = (mu_a - mu_b) - (target_a - target_b)
    loss = contrast.pow(2) + var_a + var_b
    if weight is not None:
        return (loss * weight).sum()
    return loss.mean()


def combined_expected_loss(
    mu_a: torch.Tensor,
    var_a: torch.Tensor,
    mu_b: torch.Tensor,
    var_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    lam: float = 1.0,
    *,
    point_weight: torch.Tensor | None = None,
    pair_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Separate weighted sums for the pointwise and pair terms."""

    if lam < 0:
        raise ValueError("lambda must be nonnegative")
    point = 0.5 * (
        expected_point_loss(mu_a, var_a, target_a, weight=point_weight)
        + expected_point_loss(mu_b, var_b, target_b, weight=point_weight)
    )
    pair = expected_pair_loss(
        mu_a, var_a, mu_b, var_b, target_a, target_b, weight=pair_weight
    )
    return {"point": point, "pair": pair, "total": point + lam * pair}


def mean_report_objective(
    mu_a: torch.Tensor,
    mu_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    lam: float = 1.0,
    *,
    point_weight: torch.Tensor | None = None,
    pair_weight: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """The squared-error-of-the-mean control (variance terms dropped)."""

    point = 0.5 * (
        expected_point_loss(mu_a, torch.zeros_like(mu_a), target_a, weight=point_weight)
        + expected_point_loss(mu_b, torch.zeros_like(mu_b), target_b, weight=point_weight)
    )
    pair = expected_pair_loss(
        mu_a,
        torch.zeros_like(mu_a),
        mu_b,
        torch.zeros_like(mu_b),
        target_a,
        target_b,
        weight=pair_weight,
    )
    return {"point": point, "pair": pair, "total": point + lam * pair}


def enumerate_expected_loss(
    probs_a: torch.Tensor,
    probs_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    *,
    lam: float = 1.0,
    grid: torch.Tensor | None = None,
) -> dict[str, float]:
    """Exhaustive evaluation of the expected loss over all grid pairs.

    Reference implementation for the moment formula; used by the unit test
    that requires numerical agreement with exhaustive enumeration.
    """

    if grid is None:
        grid = grid_tensor(probs_a.device, probs_a.dtype)
    values = grid
    pairs = torch.cartesian_prod(values, values)  # (G*G, 2)
    c_a = pairs[:, 0]
    c_b = pairs[:, 1]
    weight = (probs_a.unsqueeze(-1) * probs_b.unsqueeze(-2)).reshape(-1)
    point = 0.5 * ((c_a - target_a) ** 2 + (c_b - target_b) ** 2)
    pair = ((c_a - c_b) - (target_a - target_b)) ** 2
    return {
        "point": float((weight * point).sum()),
        "pair": float((weight * pair).sum()),
        "total": float((weight * (point + lam * pair)).sum()),
    }


@dataclass
class WeightedStates:
    slots: list[str]
    weights: torch.Tensor

    @classmethod
    def from_slots(cls, slots: list[str], mapping: dict | None = None) -> "WeightedStates":
        mapping = mapping or CONDITION_WEIGHTS
        weights = torch.tensor([mapping[slot] for slot in slots], dtype=torch.float32)
        weights = weights / weights.sum()
        return cls(slots=slots, weights=weights)


def pair_type_weights(
    pair_slots: list[tuple[str, str]], pair_types: tuple = PAIR_TYPES
) -> torch.Tensor:
    """Equal weight over prespecified pair types, split within each type.

    A pair whose slot combination is not prespecified receives zero weight;
    this keeps the aggregate contrast term over the declared family only.
    """

    counts: dict[tuple[str, str], int] = {}
    for pair in pair_slots:
        counts[pair] = counts.get(pair, 0) + 1
    weights = torch.zeros(len(pair_slots), dtype=torch.float32)
    for index, pair in enumerate(pair_slots):
        if pair in counts and pair in pair_types:
            weights[index] = 1.0 / len(pair_types) / counts[pair]
    return weights


# ---------------------------------------------------------------------------
# Sampled reward estimator (T6 comparator)
# ---------------------------------------------------------------------------
def point_reward(c_a: torch.Tensor, c_b: torch.Tensor, target_a: torch.Tensor, target_b: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((c_a - target_a).pow(2) + (c_b - target_b).pow(2))


def pair_reward(c_a: torch.Tensor, c_b: torch.Tensor, target_a: torch.Tensor, target_b: torch.Tensor) -> torch.Tensor:
    return -((c_a - c_b) - (target_a - target_b)).pow(2)


def joint_reward(
    c_a: torch.Tensor,
    c_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    lam: float = 1.0,
) -> torch.Tensor:
    return point_reward(c_a, c_b, target_a, target_b) + lam * pair_reward(
        c_a, c_b, target_a, target_b
    )


def leave_one_out_baseline(rewards: torch.Tensor) -> torch.Tensor:
    """Baseline from the other samples in the group (independent of the action)."""

    if rewards.ndim != 2:
        raise ValueError("rewards must be (n_states, group_size)")
    total = rewards.sum(dim=1, keepdim=True)
    group = rewards.shape[1]
    if group < 2:
        return torch.zeros_like(rewards)
    return (total - rewards) / (group - 1)


def reinforce_loss(
    log_probs: torch.Tensor,
    rewards: torch.Tensor,
    *,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Plain REINFORCE with an optional leave-one-out baseline.

    ``log_probs`` and ``rewards`` have shape ``(n_samples,)``.  The plan says
    any baseline must be independent of the sampled action conditional on the
    state; ``leave_one_out_baseline`` satisfies this for grouped samples.
    """

    advantage = rewards if baseline is None else rewards - baseline
    return -(log_probs * advantage.detach()).mean()


def kl_to_reference(probs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Exact categorical KL per row, ``KL(probs || reference)``."""

    return (probs * ((probs + EPS).log() - (reference + EPS).log())).sum(dim=-1)


def default_lambda(protocol=PROTOCOL) -> float:
    return protocol.pair_lambda
