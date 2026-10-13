"""Training arms and the alternating retrieval/reporting procedure.

Gradient paths (proposal section 6.5):

===============  ==================  =================
signal           retrieval adapters  reporting adapter
===============  ==================  =================
correctness RL   yes                 no
selectivity      yes                 no
proper score     no                  yes
===============  ==================  =================

The toy's answer space is small, so the correctness reward is available in
closed form, ``E[Y] = pi[a*]``.  The exact expected-reward gradient has the
same expectation as the sampled policy gradient but no estimator variance;
the sampled path is kept in ``sampled_reward`` for an optional check.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .config import REPORT_CONDITIONS, ToyConfig
from .model import (
    Selector,
    set_all_trainable,
    set_report_trainable,
    set_retrieval_trainable,
)
from .world import World

EPS = 1e-9


def kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Exact categorical KL per row, ``KL(p || q)``."""

    return (p * ((p + EPS).log() - (q + EPS).log())).sum(dim=-1)


def sample_batch(
    config: ToyConfig,
    subjects_pool: torch.Tensor,
    generator: torch.Generator,
    device: torch.device | str | None = None,
):
    subjects = subjects_pool[
        torch.randint(len(subjects_pool), (config.batch,), generator=generator)
    ]
    relations = torch.randint(config.n_relations, (config.batch,), generator=generator)
    if device is not None:
        subjects = subjects.to(device)
        relations = relations.to(device)
    return subjects, relations


def _bindings(world: World, generator: torch.Generator, step: int, offset: int) -> torch.Tensor:
    """Fresh bindings for the training batch (proposal section 8.1).

    Resampling every step prevents memorizing the fixed subject/relation table;
    the model must read the relation component from the memory.
    """

    return world.resample_assignments(int(generator.initial_seed()) + offset + step)


def warm_start(config: ToyConfig, world: World, model: Selector, seed: int) -> Selector:
    """Shared supervised warm start on clean bindings (proposal section 8.1)."""

    generator = torch.Generator().manual_seed(seed)
    set_all_trainable(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.warm_start_lr)
    model.train()
    for step in range(config.warm_start_steps):
        subjects, relations = sample_batch(config, world.train_ids, generator, world.device)
        assignments = _bindings(world, generator, step, 10_000)
        memory = world.clean_memory(subjects, assignments)
        correct = assignments[subjects, relations]
        logits = model.answer_logits(memory, relations)
        loss = F.cross_entropy(logits, correct)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def _reference_policy(reference: Selector, memory: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return reference.policy(memory, relations)


def train_retrieval(
    config: ToyConfig,
    world: World,
    model: Selector,
    arm: str,
    seed: int,
    reference: Selector | None = None,
) -> Selector:
    """One retrieval-training arm: ``B`` correctness, ``C`` hybrid, ``D`` supervised."""

    if arm not in ("B", "C", "D"):
        raise ValueError(f"unknown retrieval arm: {arm}")
    reference = reference if reference is not None else copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    generator = torch.Generator().manual_seed(seed)
    set_retrieval_trainable(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    model.train()

    for step in range(config.steps):
        subjects, relations = sample_batch(config, world.train_ids, generator, world.device)
        assignments = _bindings(world, generator, step, 20_000)
        correct = assignments[subjects, relations]
        clean_memory = world.clean_memory(subjects, assignments)
        clean_logits = model.answer_logits(clean_memory, relations)
        clean_policy = torch.softmax(clean_logits / config.temperature, dim=-1)
        reference_policy = _reference_policy(reference, clean_memory, relations)
        reference_kl = kl_divergence(clean_policy, reference_policy).mean()

        if arm == "B":
            reward = clean_policy.gather(1, correct[:, None]).squeeze(1).mean()
            loss = -reward + config.beta_kl * reference_kl
        else:
            swapped_memory, _ = world.condition(
                subjects,
                relations,
                "irrelevant_swap",
                seed=int(generator.initial_seed()) + step,
                assignments=assignments,
            )
            swapped_logits = model.answer_logits(swapped_memory, relations)
            swapped_policy = torch.softmax(swapped_logits / config.temperature, dim=-1)
            ce_clean = -clean_policy.gather(1, correct[:, None]).squeeze(1).clamp_min(EPS).log()
            ce_swapped = -swapped_policy.gather(1, correct[:, None]).squeeze(1).clamp_min(EPS).log()
            consistency = kl_divergence(clean_policy.detach(), swapped_policy)
            selectivity = ce_clean + ce_swapped + config.eta_consistency * consistency
            if arm == "C":
                reward = clean_policy.gather(1, correct[:, None]).squeeze(1).mean()
                loss = (
                    -reward
                    + config.beta_kl * reference_kl
                    + config.lambda_sel * selectivity.mean()
                )
            else:  # arm D: supervised retrieval plus the same selectivity loss
                loss = selectivity.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def sampled_reward(
    model: Selector,
    memory: torch.Tensor,
    relations: torch.Tensor,
    correct: torch.Tensor,
    *,
    group_size: int = 8,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampled policy-gradient rewards with a leave-one-out baseline.

    Returns ``(log_probs, advantages)`` for the sampled answers.  This mirrors
    a GRPO-style estimator; the primary toy experiments use the exact expected
    reward instead.
    """

    policy = model.policy(memory, relations)
    sample = torch.multinomial(
        policy.detach().cpu(), group_size, replacement=True, generator=generator
    ).to(policy.device)
    rewards = (sample == correct[:, None]).float()
    total = rewards.sum(dim=1, keepdim=True)
    baseline = (total - rewards) / max(group_size - 1, 1)
    advantages = rewards - baseline
    log_probs = policy.log().gather(1, sample)
    return log_probs, advantages


@torch.no_grad()
def sample_outcomes(
    policy: torch.Tensor,
    correct: torch.Tensor,
    outcomes: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Independent correctness outcomes under the frozen answer policy."""

    answers = torch.multinomial(
        policy.detach().cpu(), outcomes, replacement=True, generator=generator
    ).to(correct.device)
    return (answers == correct[:, None]).float()


def train_report(config: ToyConfig, world: World, model: Selector, seed: int) -> Selector:
    """Train the report head against frozen-policy outcomes (proposal section 6.3)."""

    generator = torch.Generator().manual_seed(seed)
    set_report_trainable(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.report_lr)
    grid = torch.tensor(
        config.confidence_grid, dtype=torch.float32, device=next(model.parameters()).device
    )
    model.eval()

    for step in range(config.report_steps):
        subjects, relations = sample_batch(config, world.train_ids, generator, world.device)
        assignments = _bindings(world, generator, step, 30_000)
        name = REPORT_CONDITIONS[int(torch.randint(len(REPORT_CONDITIONS), (1,), generator=generator))]
        memory, correct = world.condition(
            subjects,
            relations,
            name,
            seed=int(generator.initial_seed()) + step,
            assignments=assignments,
        )
        with torch.no_grad():
            policy = model.policy(memory, relations)
            outcomes = sample_outcomes(
                policy, correct, config.outcomes_per_item, generator
            )
            target = outcomes.mean(dim=1)
        report = model.report_distribution(memory, relations)
        loss = (report * (grid[None, :] - target[:, None]) ** 2).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def run_arm(config: ToyConfig, world: World, model: Selector, arm: str, seed: int) -> Selector:
    """Alternating procedure for the E2 arms (proposal section 6.6)."""

    reference = copy.deepcopy(model).eval()
    for round_index in range(config.rounds):
        if arm != "A":
            train_retrieval(config, world, model, arm, seed + round_index, reference)
        train_report(config, world, model, seed + 100 + round_index)
    return model
