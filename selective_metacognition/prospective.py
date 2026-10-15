"""E3: prospective reporting before a specified intervention.

The reporter sees the **clean** state and a description of an intervention
that will be applied *after* the report.  It predicts success under the
post-intervention policy.  The ordinary baseline receives the clean
output-probability features (max probability, entropy, top-2 margin) and the
same intervention description, but not the reporter.  Neither predictor sees
the post-intervention state.

This makes the incremental endpoint meaningful where E2's could not be: in E2
the baseline saw the post-intervention probabilities, which determine success
by construction.  Here the baseline must extrapolate from the clean state, and
the internal state may carry item-specific fragility that the clean output
probabilities do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import (
    PROSPECTIVE_HELDOUT_SCALES,
    PROSPECTIVE_HELDOUT_STRENGTHS,
    PROSPECTIVE_TRAIN_SCALES,
    PROSPECTIVE_TRAIN_STRENGTHS,
    PROSPECTIVE_TYPES,
    ToyConfig,
)
from .evaluation import all_pairs
from .model import Selector, set_report_trainable
from .training import _bindings, sample_batch, sample_outcomes
from .world import World

SWAP_SCALE_REFERENCE = 3.0


@dataclass(frozen=True)
class ProspectiveSpec:
    name: str
    parameter: float

    @property
    def normalized_parameter(self) -> float:
        """Map the parameter to a comparable scale in [0, 1]."""

        if self.name == "irrelevant_swap":
            return self.parameter / SWAP_SCALE_REFERENCE
        return self.parameter


def training_specs() -> list[ProspectiveSpec]:
    specs = [
        ProspectiveSpec("target_weaken", strength) for strength in PROSPECTIVE_TRAIN_STRENGTHS
    ]
    specs += [
        ProspectiveSpec("irrelevant_weaken", strength) for strength in PROSPECTIVE_TRAIN_STRENGTHS
    ]
    specs += [ProspectiveSpec("irrelevant_swap", scale) for scale in PROSPECTIVE_TRAIN_SCALES]
    return specs


def heldout_specs() -> list[ProspectiveSpec]:
    specs = [
        ProspectiveSpec("target_weaken", strength) for strength in PROSPECTIVE_HELDOUT_STRENGTHS
    ]
    specs += [
        ProspectiveSpec("irrelevant_weaken", strength)
        for strength in PROSPECTIVE_HELDOUT_STRENGTHS
    ]
    specs += [ProspectiveSpec("irrelevant_swap", scale) for scale in PROSPECTIVE_HELDOUT_SCALES]
    return specs


def intervention_context(spec: ProspectiveSpec, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """One-hot intervention type plus the normalized parameter."""

    context = torch.zeros(len(PROSPECTIVE_TYPES) + 1, device=device, dtype=dtype)
    context[PROSPECTIVE_TYPES.index(spec.name)] = 1.0
    context[-1] = spec.normalized_parameter
    return context


def apply_spec(
    world: World,
    subjects: torch.Tensor,
    relations: torch.Tensor,
    spec: ProspectiveSpec,
    assignments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the post-intervention memory and the (unchanged) correct answer."""

    if spec.name == "irrelevant_swap":
        return world.condition(
            subjects,
            relations,
            spec.name,
            swap_scale=spec.parameter,
            assignments=assignments,
        )
    return world.condition(
        subjects,
        relations,
        spec.name,
        strength=spec.parameter,
        assignments=assignments,
    )


@torch.no_grad()
def collect_prospective_rows(
    model: Selector,
    world: World,
    subject_ids: torch.Tensor,
    *,
    assignments: torch.Tensor | None = None,
    specs: list[ProspectiveSpec] | None = None,
    outcomes: int = 8,
    seed: int = 0,
) -> list[dict]:
    """One row per (subject, relation, intervention) with fresh post outcomes."""

    specs = specs or training_specs()
    subjects, relations = all_pairs(subject_ids, world.config.n_relations)
    generator = torch.Generator().manual_seed(seed)
    rows: list[dict] = []
    for index, spec in enumerate(specs):
        clean_memory, correct = world.condition(
            subjects, relations, "clean", assignments=assignments, seed=seed + 13 * index
        )
        post_memory, _ = apply_spec(world, subjects, relations, spec, assignments)
        relations = relations.to(clean_memory.device)
        clean_policy = model.policy(clean_memory, relations)
        post_policy = model.policy(post_memory, relations)
        p_post = post_policy.gather(1, correct[:, None]).squeeze(1)
        top2 = clean_policy.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        context = intervention_context(spec, clean_memory.device, clean_memory.dtype)
        context = context.unsqueeze(0).expand(len(subjects), -1)
        report = model.expected_report(clean_memory, relations, context)
        answers = torch.multinomial(
            post_policy.detach().cpu(), outcomes, replacement=True, generator=generator
        ).to(correct.device)
        outcomes_tensor = (answers == correct[:, None]).float()
        for row_index in range(len(subjects)):
            rows.append(
                {
                    "subject_id": int(subjects[row_index]),
                    "relation": int(relations[row_index]),
                    "intervention": spec.name,
                    "strength": spec.normalized_parameter,
                    "p_correct": float(p_post[row_index]),
                    "report": float(report[row_index]),
                    "max_prob": float(clean_policy[row_index].max()),
                    "entropy": float(
                        -(clean_policy[row_index] * (clean_policy[row_index] + 1e-9).log()).sum()
                    ),
                    "margin": float(margin[row_index]),
                    "outcomes": outcomes_tensor[row_index].tolist(),
                }
            )
    return rows


def train_prospective_report(
    config: ToyConfig, world: World, model: Selector, seed: int
) -> Selector:
    """Rollout store of (clean hidden, intervention context) -> post outcomes."""

    generator = torch.Generator().manual_seed(seed)
    set_report_trainable(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.report_lr)
    grid = torch.tensor(
        config.confidence_grid, dtype=torch.float32, device=next(model.parameters()).device
    )
    specs = training_specs()
    model.eval()

    features: list[torch.Tensor] = []
    contexts: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    with torch.no_grad():
        for step in range(config.report_collect_steps):
            subjects, relations = sample_batch(config, world.train_ids, generator, world.device)
            assignments = _bindings(world, generator, step, 80_000)
            spec = specs[int(torch.randint(len(specs), (1,), generator=generator))]
            clean_memory, correct = world.condition(
                subjects, relations, "clean", assignments=assignments
            )
            post_memory, _ = apply_spec(world, subjects, relations, spec, assignments)
            post_policy = model.policy(post_memory, relations)
            features.append(model.hidden(clean_memory, relations))
            contexts.append(
                intervention_context(spec, world.device, clean_memory.dtype)
                .unsqueeze(0)
                .expand(len(subjects), -1)
            )
            outcomes = sample_outcomes(post_policy, correct, config.outcomes_per_item, generator)
            targets.append(outcomes.mean(dim=1))
    features = torch.cat(features)
    contexts = torch.cat(contexts)
    targets = torch.cat(targets)

    for _ in range(config.report_steps):
        index = torch.randint(0, len(features), (config.report_batch,), generator=generator)
        report = model.report_distribution_from_hidden(features[index], contexts[index])
        expected = (report * grid[None, :]).sum(dim=-1)
        loss = ((expected - targets[index]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model
