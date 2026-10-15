"""Reporting-branch ablation, patching, and specificity controls.

P5 asks whether a particular pathway contributes to the trained reporting
improvement (plan section 10, P5).  All interventions here are applied to a
*report-branch-only* prefix state: the report branch is built by rerunning the
frozen prefix with the intervention active, while the answer branch keeps its
original retained state and cache.  This isolation is an experimental
construction; it is not evidence that the intact model naturally separates the
two functions.

The primitives are deliberately small:

- ``report_branch_prefix``: rerun the prefix with declared hook specs and
  return a private state for the report branch.
- ``capture_group_values``: clean values of a group from a reference state.
- ``mechanism_report``: produce a greedy constrained report (or grid
  probabilities) from a report-branch state with adapters enabled.
- ``permuted_code_strings``: label-permuted but numerically equivalent output
  codes, the secondary format control.

Answer-branch logits must be checked unchanged by the caller with
``answer_branch_snapshot``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .episodes import PrefixRun, continue_teacher_forced, run_prefix
from .interventions import (
    ChannelGroup,
    InterventionHooks,
    capture_group_activations,
)
from .reporters import (
    GreedyReport,
    adapters_enabled,
    greedy_constrained_report,
    grid_logprobs,
)
from .protocol import REPORT_STRINGS


@dataclass
class ReportBranch:
    """A private report-branch state plus the hook specification used."""

    prefix: PrefixRun
    hook_specs: list
    positions: list
    perturbation_norms: dict

    def to_json(self) -> dict:
        return {
            "hooks": [
                {
                    "group": group.to_json(),
                    "kind": kind,
                }
                for group, kind, _payload in self.hook_specs
            ],
            "positions": self.positions,
            "perturbation_norms": self.perturbation_norms,
        }


def report_branch_prefix(
    model,
    input_ids: torch.Tensor,
    *,
    hook_specs: list | None = None,
    positions=None,
    attention_mask: torch.Tensor | None = None,
    prompt_hash_value: str = "",
) -> ReportBranch:
    """Rerun the prefix for the report branch only.

    ``hook_specs`` follows :class:`InterventionHooks`.  The payload for a
    ``restore``/``reinsert`` spec may be a ``(n_positions, n_channels)`` tensor
    captured from a reference state.
    """

    specs = list(hook_specs or [])
    hooks = InterventionHooks(model, specs, positions, record_norms=True)
    prefix = run_prefix(
        model,
        input_ids,
        attention_mask=attention_mask,
        hooks=hooks,
        capture_hidden=True,
        prompt_hash_value=prompt_hash_value,
    )
    return ReportBranch(
        prefix=prefix,
        hook_specs=specs,
        positions=positions,
        perturbation_norms=prefix.perturbation_norms,
    )


def capture_group_values(
    model,
    input_ids: torch.Tensor,
    group: ChannelGroup,
    positions,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return capture_group_activations(
        model, input_ids, group, positions, attention_mask=attention_mask
    )


def mechanism_report(
    model,
    branch: ReportBranch,
    suffix_ids: torch.Tensor,
    tokenizer,
    adapters,
    *,
    mode: str = "greedy",
) -> dict:
    """Produce the report from a report-branch state with adapters enabled."""

    if mode not in ("greedy", "distribution"):
        raise ValueError(f"unknown report mode: {mode!r}")
    with adapters_enabled(adapters, True):
        if mode == "greedy":
            greedy: GreedyReport = greedy_constrained_report(
                model, branch.prefix, suffix_ids, tokenizer
            )
            return {
                "string": greedy.string,
                "value": greedy.value,
                "valid": greedy.valid,
                "token_ids": greedy.token_ids,
            }
        distribution = grid_logprobs(
            model, branch.prefix, suffix_ids, tokenizer
        )
        return distribution


def reference_report(
    model,
    prefix: PrefixRun,
    suffix_ids: torch.Tensor,
    tokenizer,
    adapters,
    *,
    mode: str = "greedy",
) -> dict:
    """Report from an unmodified retained state (the intact-construction control)."""

    branch = ReportBranch(prefix=prefix, hook_specs=[], positions=None, perturbation_norms={})
    return mechanism_report(model, branch, suffix_ids, tokenizer, adapters, mode=mode)


def answer_branch_snapshot(model, prefix: PrefixRun, suffix_ids: torch.Tensor) -> torch.Tensor:
    """Answer-branch logits used to verify that report interventions do not leak."""

    return continue_teacher_forced(model, prefix, suffix_ids).detach().clone()


def permuted_code_strings(seed: int = 0) -> tuple[str, ...]:
    """A permutation of the report grammar strings.

    The mapping is deterministic and contains the same numeric strings; only
    the code positions change, which tests reliance on confidence-word or
    position semantics rather than access to reliability information.
    """

    values = list(REPORT_STRINGS)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(values), generator=generator).tolist()
    return tuple(values[index] for index in order)


def mechanism_effects(
    intact: dict,
    removed: dict,
    restored: dict,
    *,
    removal_metric=lambda report: float("nan"),
) -> dict:
    """Assemble removal/restoration effects from three report records.

    The caller supplies the metric (for example, negative Brier on held-out
    episodes).  Absolute changes are reported alongside recovery fractions so
    that an unstable ratio with a small denominator is visible (plan P5).
    """

    intact_value = removal_metric(intact)
    removed_value = removal_metric(removed)
    restored_value = removal_metric(restored)
    lost = intact_value - removed_value
    recovered = restored_value - removed_value
    recovery_fraction = recovered / lost if lost not in (0.0,) else None
    return {
        "intact": intact_value,
        "removed": removed_value,
        "restored": restored_value,
        "lost": lost,
        "recovered": recovered,
        "recovery_fraction": recovery_fraction,
    }
