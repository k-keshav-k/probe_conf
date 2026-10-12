"""Relation interventions for the retrieval-monitoring plan.

The candidate intervention fits a regularized linear map from subject states to
answer embeddings for each relation.  Let ``P_r`` project onto the learned
map's row space.  The attenuation is::

    h' = h - alpha * P_r (h - mu)

with ``mu`` the mean training subject state.  Rank and ``alpha`` are selected
on development data.  Random controls match the rank and the realized
perturbation norm.  The causal selectivity of the intervention must be
measured, not assumed from decoding accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RelationMap:
    """A learned relation direction set plus its training mean."""

    relation: str
    mean: torch.Tensor  # (d_model,)
    basis: torch.Tensor  # (rank, d_model), orthonormal rows
    singular_values: torch.Tensor  # (rank,)

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])


def fit_relation_map(
    subject_states: torch.Tensor,
    answer_embeddings: torch.Tensor,
    relation: str,
    rank: int,
    ridge: float = 1e-3,
) -> RelationMap:
    """Ridge map from subject states to answer embeddings.

    ``subject_states`` has shape ``(n, d_model)`` and ``answer_embeddings`` has
    shape ``(n, d_out)``.  The returned basis spans the row space of the fitted
    map, truncated to ``rank`` directions.
    """

    states = subject_states.float()
    targets = answer_embeddings.float()
    if states.shape[0] != targets.shape[0]:
        raise ValueError("states and targets must have the same number of rows")
    gram = states.T @ states + ridge * torch.eye(states.shape[1], device=states.device)
    weights = torch.linalg.solve(gram, states.T @ targets).T  # (d_out, d_model)
    _, singular, vh = torch.linalg.svd(weights, full_matrices=False)
    effective_rank = min(rank, int(vh.shape[0]))
    return RelationMap(
        relation=relation,
        mean=states.mean(dim=0).detach().clone(),
        basis=vh[:effective_rank].detach().clone(),
        singular_values=singular[:effective_rank].detach().clone(),
    )


def projection(hidden: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection of ``hidden`` onto the span of ``basis`` rows."""

    return (hidden @ basis.T) @ basis


def attenuation_delta(hidden: torch.Tensor, relation_map: RelationMap, alpha: float) -> torch.Tensor:
    """The vector subtracted from ``hidden`` by the attenuation."""

    centered = hidden - relation_map.mean.to(hidden.device, hidden.dtype)
    return alpha * projection(centered, relation_map.basis.to(hidden.device, hidden.dtype))


def attenuate_state(hidden: torch.Tensor, relation_map: RelationMap, alpha: float) -> torch.Tensor:
    return hidden - attenuation_delta(hidden, relation_map, alpha)


def random_map_like(relation_map: RelationMap, seed: int) -> RelationMap:
    """Random orthonormal basis with the same rank and mean."""

    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(
        relation_map.rank,
        relation_map.basis.shape[1],
        generator=generator,
        device=relation_map.basis.device,
        dtype=relation_map.basis.dtype,
    )
    q, _ = torch.linalg.qr(raw.T)
    return RelationMap(
        relation=f"random_{seed}",
        mean=relation_map.mean.clone(),
        basis=q.T.contiguous(),
        singular_values=torch.ones(relation_map.rank),
    )


def matched_random_delta(
    hidden: torch.Tensor,
    relation_map: RelationMap,
    alpha: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random perturbation with matched rank and realized norm.

    Returns the delta vector and the random map used.  The norm of the delta
    equals the norm of the relevant attenuation for this hidden state.
    """

    target = attenuation_delta(hidden, relation_map, alpha)
    random_map = random_map_like(relation_map, seed)
    candidate = attenuation_delta(hidden, random_map, 1.0)
    candidate_norm = candidate.norm()
    if float(candidate_norm) <= 1e-12:
        return torch.zeros_like(target), random_map
    scaled = candidate * (target.norm() / candidate_norm)
    return scaled, random_map


def attenuation_transform(
    relation_map: RelationMap,
    alpha: float,
    token_indices: list[int] | range | slice,
):
    """Forward-hook transform that attenuates one relation at given tokens."""

    indices = list(token_indices)

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.clone()
        for index in indices:
            state = hidden[:, index, :]
            hidden[:, index, :] = attenuate_state(state, relation_map, alpha)
        return hidden

    return transform


def cross_states(
    hidden: torch.Tensor,
    relevant_map: RelationMap,
    other_map: RelationMap,
    alpha: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Four paired states for one subject and one queried relation.

    States: ``clean``, ``relevant`` (queried-relation map), ``other`` (the
    other relation's map), and ``random`` (rank/norm-matched random control).
    """

    random_delta, _ = matched_random_delta(hidden, relevant_map, alpha, seed)
    return {
        "clean": hidden.clone(),
        "relevant": attenuate_state(hidden, relevant_map, alpha),
        "other": attenuate_state(hidden, other_map, alpha),
        "random": hidden - random_delta,
    }


def shuffle_state_assignments(states: dict[str, torch.Tensor], seed: int) -> dict[str, torch.Tensor]:
    """Relabel states so the condition identity is concealed from the report.

    The caller keeps the true assignment for analysis.  This helper only
    returns the same tensors under shuffled keys.
    """

    generator = torch.Generator().manual_seed(seed)
    names = list(states)
    order = torch.randperm(len(names), generator=generator).tolist()
    shuffled = {}
    for position, name in enumerate(names):
        shuffled[name] = states[names[order[position]]]
    return shuffled
