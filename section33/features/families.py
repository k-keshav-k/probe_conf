"""Frozen MLP feature families for the Section 3.3 experiment.

Four families are frozen in plan section 3.2, plus the pilot's reference-free
summaries as a replication family:

- ``w2_literal``                literal ``x . v_j`` similarities
- ``gate_selection``            ``a``, ``SiLU(a)``, ``s`` and their summaries
- ``contribution_concentration`` distribution of ``e_j = |s_j| ||v_j||``
- ``output_direction``          ``||o||``, ``||o||/||x||``, ``cos(o, x)``, cancellation
- ``pilot12``                   the twelve reference-free historical summaries

Every builder returns PyTorch tensors with the same leading shape as the input
so callers can evaluate a whole batch of token positions at once.  The scalar
helpers ``build_family`` and ``build_all`` convert to Python floats for tests
and one-token use.  Reference-dependent names from the pilot
(``cos_sel_agold``, ``proj_gold_norm``, ``mlp_margin``, ``ent_cand``,
``logit_mlp_gold``) are never produced by these families.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import mlp_math as mm


@dataclass(frozen=True)
class FamilySpec:
    name: str
    uses_reference: bool
    description: str


FAMILY_REGISTRY: dict[str, FamilySpec] = {
    "w2_literal": FamilySpec(
        "w2_literal",
        False,
        "Literal x . v_j similarities and cosine similarities with norm controls.",
    ),
    "gate_selection": FamilySpec(
        "gate_selection",
        False,
        "Gate input a, SiLU(a), selection s, and their signed distribution summaries.",
    ),
    "contribution_concentration": FamilySpec(
        "contribution_concentration",
        False,
        "Distribution of e_j = |s_j| ||v_j||: entropy, top masses, effective channels.",
    ),
    "output_direction": FamilySpec(
        "output_direction",
        False,
        "Norm, relative norm, cosine with the input, and cancellation of the write.",
    ),
    "pilot12": FamilySpec(
        "pilot12",
        False,
        "The pilot's twelve reference-free summary statistics (replication family).",
    ),
}

#: Names that use the reference answer and must not enter confirmatory probes.
REFERENCE_DEPENDENT_NAMES = (
    "cos_sel_agold",
    "proj_gold_norm",
    "mlp_margin",
    "ent_cand",
    "logit_mlp_gold",
)


# ---------------------------------------------------------------------------
# Family builders (tensor output, leading dimensions preserved)
# ---------------------------------------------------------------------------
def build_w2_literal(t: mm.MLPTensors) -> dict[str, torch.Tensor]:
    """Literal ``x . v_j`` and cosine similarities (no softmax conversion)."""

    assert t.w2_dot is not None, "w2_literal needs the x @ W_down scores"
    dot = t.w2_dot
    v_norms = t.v_norms
    x_norm = t.x.norm(dim=-1, keepdim=True)
    cosine = dot / (x_norm * v_norms + mm.EPS)

    out: dict[str, torch.Tensor] = {}
    for key, value in mm.signed_summary(dot).items():
        out[f"dot_{key}"] = value
    out["dot_l2"] = dot.norm(dim=-1)
    out["cos_max"] = cosine.max(dim=-1).values
    out["cos_mean"] = cosine.mean(dim=-1)
    out["cos_min"] = cosine.min(dim=-1).values
    # Concentration of the absolute literal similarity, clearly named as the
    # literal family so it is not confused with the selection family.
    abs_dot = dot.abs()
    for key, value in mm.concentration_summary(abs_dot).items():
        out[f"absdot_{key}"] = value
    return out


def build_gate_selection(t: mm.MLPTensors) -> dict[str, torch.Tensor]:
    """``a``, ``SiLU(a)``, and the actual selection ``s``."""

    out: dict[str, torch.Tensor] = {}
    for key, value in mm.signed_summary(t.a).items():
        out[f"a_{key}"] = value
    for key, value in mm.signed_summary(t.s).items():
        out[f"s_{key}"] = value
    out["a_l2"] = t.a.norm(dim=-1)
    out["s_l2"] = t.s.norm(dim=-1)
    out["silu_a_l2"] = torch.nn.functional.silu(t.a).norm(dim=-1)
    for key, value in mm.concentration_summary(t.a.abs()).items():
        out[f"absa_{key}"] = value
    for key, value in mm.concentration_summary(t.s.abs()).items():
        out[f"abss_{key}"] = value
    return out


def build_contribution_concentration(t: mm.MLPTensors) -> dict[str, torch.Tensor]:
    """Distribution of channel contribution magnitudes ``e_j``."""

    e = mm.contribution_magnitudes(t.s, t.v_norms)
    out = {f"e_{key}": value for key, value in mm.concentration_summary(e).items()}
    out["e_total"] = e.sum(dim=-1)
    out["e_max"] = e.max(dim=-1).values
    out["v_norm_mean"] = t.v_norms.mean().expand_as(out["e_total"])
    out["v_norm_std"] = t.v_norms.std().expand_as(out["e_total"])
    return out


def build_output_direction(t: mm.MLPTensors) -> dict[str, torch.Tensor]:
    return {f"o_{key}": value for key, value in mm.direction_summary(t).items()}


def build_pilot12(t: mm.MLPTensors) -> dict[str, torch.Tensor]:
    """Exact reference-free summaries from the pilot collector."""

    a, s, o, h = t.a, t.s, t.o, t.h_post
    return {
        "ent_gate": mm.softmax_entropy(a),
        "ent_gate_sq": mm.entropy_of_squares(a),
        "active_frac": (a > 0).to(a.dtype).mean(dim=-1),
        "top5_gate": mm.top_k_mass(a * a, 5),
        "gini_gate": mm.gini(a),
        "max_gate": a.max(dim=-1).values,
        "kurt_gate": mm.kurtosis(a),
        "ent_sel_sq": mm.entropy_of_squares(s),
        "l2_sel": s.norm(dim=-1),
        "l2_down": o.norm(dim=-1),
        "down_norm_ratio": o.norm(dim=-1) / (h.norm(dim=-1) + mm.EPS),
        "cos_down_h": (o * h).sum(dim=-1) / (o.norm(dim=-1) * h.norm(dim=-1) + mm.EPS),
    }


_BUILDERS = {
    "w2_literal": build_w2_literal,
    "gate_selection": build_gate_selection,
    "contribution_concentration": build_contribution_concentration,
    "output_direction": build_output_direction,
    "pilot12": build_pilot12,
}


def build_family_tensors(name: str, tensors: mm.MLPTensors) -> dict[str, torch.Tensor]:
    if name not in _BUILDERS:
        raise KeyError(f"unknown feature family: {name}")
    spec = FAMILY_REGISTRY[name]
    if spec.uses_reference:
        raise ValueError(f"family {name} is not reference-free")
    with torch.no_grad():
        return _BUILDERS[name](tensors)


def build_family(name: str, tensors: mm.MLPTensors) -> dict[str, float]:
    """Scalar helper: one value per statistic (input must have one row)."""

    stats = build_family_tensors(name, tensors)
    out: dict[str, float] = {}
    for key, value in stats.items():
        if value.numel() != 1:
            raise ValueError(
                f"build_family expects a single row; {name}.{key} has shape {tuple(value.shape)}. "
                "Use build_family_tensors for batches."
            )
        out[key] = float(value)
    return out


def build_all_tensors(
    tensors: mm.MLPTensors, families: tuple[str, ...] | None = None
) -> dict[str, torch.Tensor]:
    """All requested families for a batch of rows, one tensor per statistic."""

    selected = families or tuple(FAMILY_REGISTRY)
    out: dict[str, torch.Tensor] = {}
    for family in selected:
        for stat, value in build_family_tensors(family, tensors).items():
            out[f"{family}__{stat}"] = value
    assert_reference_free(out.keys())
    return out


def build_all(tensors: mm.MLPTensors, families: tuple[str, ...] | None = None) -> dict[str, float]:
    """Scalar helper used by tests and one-token callers."""

    stats = build_all_tensors(tensors, families)
    out: dict[str, float] = {}
    for name, value in stats.items():
        if value.numel() != 1:
            raise ValueError(
                f"build_all expects a single row; {name} has shape {tuple(value.shape)}. "
                "Use build_all_tensors for batches."
            )
        out[name] = float(value)
    return out


def assert_reference_free(names) -> None:
    """Raise if a reference-dependent historical name appears."""

    bad = [n for n in names if any(ref in n for ref in REFERENCE_DEPENDENT_NAMES)]
    if bad:
        raise ValueError(f"reference-dependent features in a confirmatory set: {bad}")
