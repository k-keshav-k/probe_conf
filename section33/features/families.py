"""Frozen MLP feature families for the Section 3.3 experiment.

Four families are frozen in plan section 3.2, plus the pilot's reference-free
summaries as a replication family:

- ``w2_literal``                literal ``x . v_j`` similarities
- ``gate_selection``            ``a``, ``SiLU(a)``, ``s`` and their summaries
- ``contribution_concentration`` distribution of ``e_j = |s_j| ||v_j||``
- ``output_direction``          ``||o||``, ``||o||/||x||``, ``cos(o, x)``, cancellation
- ``pilot12``                   the twelve reference-free historical summaries

Every builder returns plain Python floats keyed by statistic name.  The caller
adds the layer and position tags.  Reference-dependent names from the pilot
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
# Family builders
# ---------------------------------------------------------------------------
def build_w2_literal(t: mm.MLPTensors) -> dict[str, float]:
    """Literal ``x . v_j`` and cosine similarities (no softmax conversion)."""

    assert t.w2_dot is not None, "w2_literal needs the x @ W_down scores"
    dot = t.w2_dot
    v_norms = t.v_norms
    x_norm = t.x.norm(dim=-1, keepdim=True)
    cosine = dot / (x_norm * v_norms + mm.EPS)

    out: dict[str, float] = {}
    for key, value in mm.signed_summary(dot).items():
        out[f"dot_{key}"] = float(value)
    out["dot_l2"] = float(dot.norm(dim=-1))
    out["cos_max"] = float(cosine.max(dim=-1).values)
    out["cos_mean"] = float(cosine.mean(dim=-1))
    out["cos_min"] = float(cosine.min(dim=-1).values)
    # Concentration of the absolute literal similarity, clearly named as the
    # literal family so it is not confused with the selection family.
    abs_dot = dot.abs()
    for key, value in mm.concentration_summary(abs_dot).items():
        out[f"absdot_{key}"] = float(value)
    return out


def build_gate_selection(t: mm.MLPTensors) -> dict[str, float]:
    """``a``, ``SiLU(a)``, and the actual selection ``s``."""

    out: dict[str, float] = {}
    for key, value in mm.signed_summary(t.a).items():
        out[f"a_{key}"] = float(value)
    for key, value in mm.signed_summary(t.s).items():
        out[f"s_{key}"] = float(value)
    out["a_l2"] = float(t.a.norm(dim=-1))
    out["s_l2"] = float(t.s.norm(dim=-1))
    out["silu_a_l2"] = float(torch.nn.functional.silu(t.a).norm(dim=-1))
    for key, value in mm.concentration_summary(t.a.abs()).items():
        out[f"absa_{key}"] = float(value)
    for key, value in mm.concentration_summary(t.s.abs()).items():
        out[f"abss_{key}"] = float(value)
    return out


def build_contribution_concentration(t: mm.MLPTensors) -> dict[str, float]:
    """Distribution of channel contribution magnitudes ``e_j``."""

    e = mm.contribution_magnitudes(t.s, t.v_norms)
    out = {f"e_{key}": float(value) for key, value in mm.concentration_summary(e).items()}
    out["e_total"] = float(e.sum(dim=-1))
    out["e_max"] = float(e.max(dim=-1).values)
    out["v_norm_mean"] = float(t.v_norms.mean())
    out["v_norm_std"] = float(t.v_norms.std())
    return out


def build_output_direction(t: mm.MLPTensors) -> dict[str, float]:
    return {f"o_{key}": float(value) for key, value in mm.direction_summary(t).items()}


def build_pilot12(t: mm.MLPTensors) -> dict[str, float]:
    """Exact reference-free summaries from the pilot collector."""

    a, s, o, h = t.a, t.s, t.o, t.h_post
    out = {
        "ent_gate": float(mm.softmax_entropy(a)),
        "ent_gate_sq": float(mm.entropy_of_squares(a)),
        "active_frac": float((a > 0).float().mean(dim=-1)),
        "top5_gate": float(
            mm.top_k_mass(a * a, 5)
        ),
        "gini_gate": float(mm.gini(a)),
        "max_gate": float(a.max(dim=-1).values),
        "kurt_gate": float(mm.kurtosis(a)),
        "ent_sel_sq": float(mm.entropy_of_squares(s)),
        "l2_sel": float(s.norm(dim=-1)),
        "l2_down": float(o.norm(dim=-1)),
        "down_norm_ratio": float(o.norm(dim=-1) / (h.norm(dim=-1) + mm.EPS)),
        "cos_down_h": float(
            (o * h).sum(dim=-1) / (o.norm(dim=-1) * h.norm(dim=-1) + mm.EPS)
        ),
    }
    return out


_BUILDERS = {
    "w2_literal": build_w2_literal,
    "gate_selection": build_gate_selection,
    "contribution_concentration": build_contribution_concentration,
    "output_direction": build_output_direction,
    "pilot12": build_pilot12,
}


def build_family(name: str, tensors: mm.MLPTensors) -> dict[str, float]:
    if name not in _BUILDERS:
        raise KeyError(f"unknown feature family: {name}")
    spec = FAMILY_REGISTRY[name]
    if spec.uses_reference:
        raise ValueError(f"family {name} is not reference-free")
    with torch.no_grad():
        return _BUILDERS[name](tensors)


def build_all(tensors: mm.MLPTensors, families: tuple[str, ...] | None = None) -> dict[str, float]:
    """Build every requested family and prefix each statistic with its family."""

    selected = families or tuple(FAMILY_REGISTRY)
    out: dict[str, float] = {}
    with torch.no_grad():
        for family in selected:
            for stat, value in build_family(family, tensors).items():
                out[f"{family}__{stat}"] = value
    assert_reference_free(out.keys())
    return out


def assert_reference_free(names) -> None:
    """Raise if a reference-dependent historical name appears."""

    bad = [n for n in names if any(ref in n for ref in REFERENCE_DEPENDENT_NAMES)]
    if bad:
        raise ValueError(f"reference-dependent features in a confirmatory set: {bad}")
