"""Stage runners for P1 (episode contract) and P2 (intervention validation).

P0 lives in :mod:`causal_confidence.p0`.  P3-P6 are gated on the P2 result and
raise :class:`StageGated` until then; their contracts are implemented in the
library modules, but no training is launched here.

All stage outputs go to ``runs/causal_confidence/<protocol_version>/<run_id>/``
and never overwrite an existing frozen artifact.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import interventions as iv
from .data import Question, audit_references, leakage_scan
from .episodes import (
    adapters_enabled,
    answer_suffix_ids,
    build_prefix_ids,
    cache_full_run_equivalence,
    check_adapter_gating,
    check_prefix_immutability,
    continue_teacher_forced,
    duplicate_run_tolerance,
    generate_answer,
    last_layer_cache_miss,
    report_suffix_ids,
    run_prefix,
)
from .evaluate import ClusterItem, cluster_bootstrap, gate_decision
from .outcomes import OutcomeBank, audit_banks, sample_bank
from .protocol import PROTOCOL, Protocol
from .reporters import (
    base_parameter_snapshot,
    enable_reporter_gradients,
    freeze_answer_policy,
    install_report_adapters,
    remove_report_adapters,
    reporter_parameters,
    snapshot_matches,
)


class StageGated(RuntimeError):
    """Raised when a later stage is requested before its gate passes."""


# ---------------------------------------------------------------------------
# Model loading and prompt preparation
# ---------------------------------------------------------------------------
@dataclass
class EpisodeEnvironment:
    model: object
    tokenizer: object
    device: torch.device
    revision: str
    n_layers: int

    def prefix_ids(self, question: Question) -> tuple[torch.Tensor, str]:
        ids, digest = build_prefix_ids(self.tokenizer, question.question_text)
        return ids.to(self.device), digest

    @property
    def answer_suffix(self) -> torch.Tensor:
        return answer_suffix_ids(self.tokenizer).to(self.device)

    @property
    def report_suffix(self) -> torch.Tensor:
        return report_suffix_ids(self.tokenizer).to(self.device)


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_and_tokenizer(
    name: str = PROTOCOL.model_name,
    revision: str = PROTOCOL.model_revision,
    *,
    device: str | torch.device | None = None,
    dtype: str = "auto",
) -> EpisodeEnvironment:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = resolve_device(str(device) if device is not None else None)
    if dtype == "auto":
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, torch_dtype=torch_dtype
    )
    model.to(device)
    model.eval()
    return EpisodeEnvironment(
        model=model,
        tokenizer=tokenizer,
        device=device,
        revision=revision,
        n_layers=int(model.config.num_hidden_layers),
    )


def _last_only(ids: torch.Tensor):
    return [[int(ids.shape[1]) - 1] for _ in range(ids.shape[0])]


def bank_with_hooks(
    env: EpisodeEnvironment,
    question: Question,
    *,
    slot: str,
    hooks_specs: list,
    positions,
    draw_seed: int,
    draws: int,
    partition: str = "discovery",
    condition: dict | None = None,
) -> OutcomeBank:
    """Sample one condition bank for one question (batch size one)."""

    ids, digest = env.prefix_ids(question)
    hook_set = iv.InterventionHooks(env.model, hooks_specs, positions) if hooks_specs else None
    prefix = run_prefix(
        env.model, ids, hooks=hook_set, capture_hidden=False, prompt_hash_value=digest
    )
    return sample_bank(
        env.model,
        env.tokenizer,
        question=question,
        prefix=prefix,
        suffix_ids=env.answer_suffix,
        partition=partition,
        slot=slot,
        draw_seed=draw_seed,
        draws=draws,
        condition=condition,
        model_revision=env.revision,
    )


# ---------------------------------------------------------------------------
# P1: episode contract and answer policy
# ---------------------------------------------------------------------------
def run_p1(
    env: EpisodeEnvironment,
    questions: list[Question],
    *,
    draws: int = 8,
    adapter_layers: tuple[int, ...] | None = None,
    seed: int = PROTOCOL.inner_seed,
) -> tuple[dict, list[OutcomeBank]]:
    """Episode contract checks plus the discovery answer-policy audit."""

    model = env.model
    if not questions:
        raise ValueError("P1 needs at least one question")
    question = questions[0]
    ids, digest = env.prefix_ids(question)
    answer_suffix = env.answer_suffix
    positions = _last_only(ids)
    sites = iv.candidate_sites(env.n_layers)
    site = sites[len(sites) // 2]
    intermediate = int(model.config.intermediate_size)
    probe_channels = tuple(range(min(16, intermediate)))
    probe_group = iv.ChannelGroup(layer=site, channels=probe_channels, label="contract_probe")

    tolerance = duplicate_run_tolerance(model, ids)
    floor = cache_full_run_equivalence(
        model, ids, answer_suffix, hook_specs=[], positions=[]
    )
    middle = cache_full_run_equivalence(
        model,
        ids,
        answer_suffix,
        hook_specs=[(probe_group, "attenuate", 0.5)],
        positions=positions,
    )
    equivalence_tolerance = max(
        10.0 * floor["max_abs_diff"], 10.0 * tolerance["tolerance"], 1e-3
    )
    last_layer = last_layer_cache_miss(
        model,
        ids,
        answer_suffix,
        layer_index=env.n_layers - 1,
        positions=positions,
        channels=probe_channels,
    )

    prefix = run_prefix(model, ids, prompt_hash_value=digest)
    isolation = check_prefix_immutability(model, prefix, answer_suffix)

    # Adapter gating on the real model; removed again before the policy audit
    # so the environment stays pristine for later stages.
    layers = adapter_layers or (site,)
    adapters = install_report_adapters(model, layers, rank=PROTOCOL.report_rank)
    freeze_answer_policy(model)
    gating = check_adapter_gating(model, prefix, answer_suffix, adapters)
    snapshot = base_parameter_snapshot(model)
    with torch.no_grad():
        for adapter in adapters:
            adapter.lora_b.add_(0.05 * torch.randn_like(adapter.lora_b))
    gating_trained = check_adapter_gating(model, prefix, answer_suffix, adapters)
    with torch.no_grad():
        for adapter in adapters:
            adapter.lora_b.zero_()

    # One reporter update must leave the answer branch and every non-adapter
    # parameter unchanged within the duplicate-run tolerance.
    answer_logits_before = continue_teacher_forced(model, prefix, answer_suffix)
    for parameter in reporter_parameters(model):
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(reporter_parameters(model), lr=1e-4)
    target_ids = torch.tensor(
        [env.tokenizer.encode("0.50", add_special_tokens=False)], device=env.device
    )
    with adapters_enabled(adapters, True):
        full = torch.cat([env.report_suffix, target_ids], dim=1)
        logits = continue_teacher_forced(
            model, prefix, full, requires_grad=True
        )
        start = env.report_suffix.shape[1] - 1
        loss = torch.nn.functional.cross_entropy(
            logits[0, start : start + target_ids.shape[1]].float(),
            target_ids.squeeze(0),
        )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    answer_logits_after = continue_teacher_forced(model, prefix, answer_suffix)
    reporter_step = {
        "answer_logits_max_abs_diff": float(
            (answer_logits_before.float() - answer_logits_after.float()).abs().max().item()
        ),
        "base_parameters_unchanged": snapshot_matches(model, snapshot),
        "n_reporter_parameters": len(reporter_parameters(model)),
    }
    remove_report_adapters(model, adapters)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    # First-token sampling diagnostic (the exact distribution check is in the
    # unit tests; here we record the empirical gap on the real model).
    with torch.no_grad():
        reference_logits = continue_teacher_forced(model, prefix, answer_suffix)[0, -1]
        tokens = []
        for index in range(64):
            draw = generate_answer(
                model, prefix, answer_suffix, env.tokenizer, max_new_tokens=1, seed=seed + index
            )
            tokens.append(draw.token_ids[0] if draw.token_ids else -1)
    expected = torch.softmax(reference_logits.float(), dim=-1)
    empirical = torch.zeros_like(expected)
    for token in tokens:
        if token >= 0:
            empirical[token] += 1.0 / len(tokens)
    first_token = {
        "n": len(tokens),
        "max_abs_frequency_gap": float((empirical - expected).abs().max().item()),
    }

    # Leakage audit: no gold reference and no condition name may appear in the
    # reporter or answer prompts.  References shorter than three characters
    # are skipped to avoid accidental substring hits inside ordinary words.
    from .episodes import build_prefix_text
    from .protocol import ANSWER_SUFFIX, CONDITION_SLOTS, REPORT_SUFFIX

    leakage_texts = [
        build_prefix_text(question.question_text)
        for question in questions
    ] + [ANSWER_SUFFIX, REPORT_SUFFIX]
    forbidden = sorted(
        {
            reference
            for question in questions
            for reference in question.references
            if len(reference) >= 3
        }
        | set(CONDITION_SLOTS)
    )
    leakage = leakage_scan(leakage_texts, forbidden)

    contract = {
        "question_id": question.question_id,
        "prompt_hash": digest,
        "tolerance": tolerance,
        "no_hook_cache_floor": floor,
        "middle_layer_equivalence": middle,
        "equivalence_tolerance": equivalence_tolerance,
        "middle_layer_passed": middle["max_abs_diff"] <= equivalence_tolerance,
        "last_layer_diagnostic": last_layer,
        "prefix_isolation": isolation,
        "adapter_gating": gating,
        "adapter_gating_after_perturbation": gating_trained,
        "reporter_step": reporter_step,
        "first_token_sampling": first_token,
        "leakage": leakage,
    }
    contract["passed"] = bool(
        contract["middle_layer_passed"]
        and isolation["cache_unchanged"]
        and isolation["branch_repeat_max_abs_diff"] == 0.0
        and gating["answer_branch_stable"]
        and gating["disabled_vs_disabled_max_abs_diff"] == 0.0
        and reporter_step["answer_logits_max_abs_diff"] == 0.0
        and reporter_step["base_parameters_unchanged"]
        and gating_trained["enabled_vs_disabled_max_abs_diff"] > 0.0
        and leakage["clean"]
    )

    # Answer-policy audit over the discovery questions.
    banks: list[OutcomeBank] = []
    for index, item in enumerate(questions):
        banks.append(
            bank_with_hooks(
                env,
                item,
                slot="clean",
                hooks_specs=[],
                positions=None,
                draw_seed=seed + 1_000 + index,
                draws=draws,
            )
        )
    bank_payloads = [bank.to_json() for bank in banks]
    audit = audit_banks(bank_payloads)
    references = audit_references(questions)
    mean_success = float(np.mean([bank.success for bank in banks])) if banks else float("nan")
    result = {
        "stage": "p1",
        "model": {"name": PROTOCOL.model_name, "revision": env.revision},
        "n_questions": len(questions),
        "draws_per_question": draws,
        "contract": contract,
        "answer_policy": {
            "references": references,
            "banks": audit,
            "mean_success": mean_success,
            "not_at_floor": mean_success > 0.02,
            "not_at_ceiling": mean_success < 0.98,
            "every_item_audited": all(bank.references for bank in banks),
        },
    }
    return result, banks


# ---------------------------------------------------------------------------
# P2: bounded discovery
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    group: iv.ChannelGroup
    size: int
    screening: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"group": self.group.to_json(), "size": self.size, "screening": self.screening}


@dataclass
class FrozenIntervention:
    target: iv.ChannelGroup
    mild_attenuation: float
    strong_attenuation: float
    rescue: iv.ChannelGroup | None
    rescue_recovery_fraction: float
    random_groups: list[iv.ChannelGroup]
    norm_matched_group: iv.ChannelGroup | None
    other_group: iv.ChannelGroup | None
    discovery: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "target": self.target.to_json(),
            "mild_attenuation": self.mild_attenuation,
            "strong_attenuation": self.strong_attenuation,
            "rescue": self.rescue.to_json() if self.rescue else None,
            "rescue_recovery_fraction": self.rescue_recovery_fraction,
            "random_groups": [group.to_json() for group in self.random_groups],
            "norm_matched_group": self.norm_matched_group.to_json()
            if self.norm_matched_group
            else None,
            "other_group": self.other_group.to_json() if self.other_group else None,
            "discovery": self.discovery,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "FrozenIntervention":
        return cls(
            target=iv.ChannelGroup.from_json(payload["target"]),
            mild_attenuation=float(payload["mild_attenuation"]),
            strong_attenuation=float(payload["strong_attenuation"]),
            rescue=iv.ChannelGroup.from_json(payload["rescue"]) if payload.get("rescue") else None,
            rescue_recovery_fraction=float(payload.get("rescue_recovery_fraction", 0.0)),
            random_groups=[
                iv.ChannelGroup.from_json(group) for group in payload.get("random_groups", [])
            ],
            norm_matched_group=iv.ChannelGroup.from_json(payload["norm_matched_group"])
            if payload.get("norm_matched_group")
            else None,
            other_group=iv.ChannelGroup.from_json(payload["other_group"])
            if payload.get("other_group")
            else None,
            discovery=payload.get("discovery", {}),
        )


def _capture_mass_profiles(
    env: EpisodeEnvironment,
    questions: list[Question],
    sites: tuple[int, ...],
) -> tuple[dict[int, dict[str, torch.Tensor]], dict[int, torch.Tensor]]:
    """Mean channel-mass profiles per layer and relation (plan section 7.1)."""

    profiles: dict[int, dict[str, list]] = {site: defaultdict(list) for site in sites}
    weights: dict[int, torch.Tensor] = {}
    layers = iv._decoder_layers(env.model)
    for site in sites:
        weights[site] = iv._mlp_down(layers[site]).weight.detach()
    with torch.no_grad():
        for question in questions:
            ids, _ = env.prefix_ids(question)
            positions = _last_only(ids)
            for site in sites:
                selection = iv.capture_selection(env.model, ids, site, positions)
                mass = iv.selection_mass(selection, weights[site])
                profiles[site][question.relation].append(mass.cpu())
    merged: dict[int, dict[str, torch.Tensor]] = {}
    for site in sites:
        merged[site] = {
            relation: iv.channel_mass_profile(torch.cat(rows, dim=0))
            for relation, rows in profiles[site].items()
        }
    return merged, weights


def _global_profile(profiles: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(list(profiles.values())).mean(dim=0)


def _screen_candidate(
    env: EpisodeEnvironment,
    questions: list[Question],
    clean_success: dict[str, float],
    group: iv.ChannelGroup,
    attenuation: float,
    *,
    draws: int,
    seed: int,
) -> dict:
    damages = []
    successes: dict[str, float] = {}
    for index, question in enumerate(questions):
        ids, _ = env.prefix_ids(question)
        positions = _last_only(ids)
        bank = bank_with_hooks(
            env,
            question,
            slot="screen",
            hooks_specs=[(group, "attenuate", attenuation)],
            positions=positions,
            draw_seed=seed + index,
            draws=draws,
            condition={"group": group.to_json(), "attenuation": attenuation},
        )
        successes[question.question_id] = bank.success
        damages.append(clean_success[question.question_id] - bank.success)
    return {
        "attenuation": attenuation,
        "mean_damage": float(np.mean(damages)),
        "mean_success": float(np.mean(list(successes.values()))),
        "per_question_damage": damages,
        "per_question_success": successes,
    }


def run_p2_discovery(
    env: EpisodeEnvironment,
    questions: list[Question],
    *,
    protocol: Protocol = PROTOCOL,
    draws: int | None = None,
    seed: int = PROTOCOL.inner_seed,
) -> dict:
    """Bounded discovery search and freeze one intervention configuration."""

    config = protocol.intervention
    draws = draws or config.screening_draws
    sites = iv.candidate_sites(env.n_layers, config)
    profiles, _ = _capture_mass_profiles(env, questions, sites)
    relation_counts: dict[str, int] = defaultdict(int)
    for question in questions:
        relation_counts[question.relation] += 1

    screen_questions = questions[: config.screening_questions]
    clean_banks = {
        question.question_id: bank_with_hooks(
            env,
            question,
            slot="clean",
            hooks_specs=[],
            positions=None,
            draw_seed=seed + index,
            draws=draws,
        )
        for index, question in enumerate(screen_questions)
    }
    clean_success = {key: bank.success for key, bank in clean_banks.items()}

    candidates: list[Candidate] = []
    for site in sites:
        global_profile = _global_profile(profiles[site])
        for size in config.group_sizes:
            channels = iv.top_contribution_channels(global_profile, size)
            candidates.append(
                Candidate(
                    group=iv.ChannelGroup(
                        layer=site, channels=channels, label=f"site{site}_g{size}"
                    ),
                    size=size,
                )
            )
    candidates = candidates[: config.max_initial_candidates]

    for index, candidate in enumerate(candidates):
        candidate.screening = _screen_candidate(
            env,
            screen_questions,
            clean_success,
            candidate.group,
            config.screening_attenuation,
            draws=draws,
            seed=seed + 10_000 + index * 137,
        )

    admissible = [
        candidate
        for candidate in candidates
        if candidate.screening["mean_damage"] <= 0.6
        and candidate.screening["mean_success"] >= 0.02
    ]
    pool = admissible or candidates
    best = max(pool, key=lambda item: item.screening["mean_damage"])

    refinements: list[dict] = []
    for index, attenuation in enumerate(config.refinement_attenuations):
        refinements.append(
            _screen_candidate(
                env,
                screen_questions,
                clean_success,
                best.group,
                attenuation,
                draws=draws,
                seed=seed + 20_000 + index * 137,
            )
        )
    screened_grid = [
        {"attenuation": config.screening_attenuation, **best.screening},
        *refinements,
    ]
    reusable = [
        item
        for item in screened_grid
        if item["mean_damage"] <= 0.6 and item["mean_success"] >= 0.02
    ]
    chosen = max(reusable or screened_grid, key=lambda item: item["mean_damage"])
    strong = float(chosen["attenuation"])
    mild = float(min(0.25, strong))
    if strong <= 0.25:
        strong = float(max(config.refinement_attenuations))
    chosen_success = chosen["per_question_success"]
    damage_value = float(chosen["mean_damage"])

    # Rescue nomination: downstream group patched with clean activations while
    # the original disruption stays active.
    downstream = [site for site in sites if site > best.group.layer]
    rescue_site = downstream[0] if downstream else max(sites)
    assert rescue_site > best.group.layer or len(sites) == 1
    rescue_options: list[tuple[str, iv.ChannelGroup]] = [
        (
            "same_channels",
            iv.ChannelGroup(
                layer=rescue_site, channels=best.group.channels, label="rescue_same_channels"
            ),
        )
    ]
    global_rescue = _global_profile(profiles[rescue_site])
    top_channels = iv.top_contribution_channels(
        global_rescue, best.group.size, exclude=best.group.channels
    )
    if top_channels != best.group.channels:
        rescue_options.append(
            (
                "top_mass",
                iv.ChannelGroup(
                    layer=rescue_site, channels=top_channels, label="rescue_top_mass"
                ),
            )
        )

    rescue_results = []
    for option_index, (name, group) in enumerate(rescue_options):
        recovered = []
        for index, question in enumerate(screen_questions):
            ids, _ = env.prefix_ids(question)
            positions = _last_only(ids)
            clean_values = iv.capture_group_activations(env.model, ids, group, positions)
            bank = bank_with_hooks(
                env,
                question,
                slot="rescue",
                hooks_specs=[
                    (best.group, "attenuate", strong),
                    (group, "restore", clean_values),
                ],
                positions=positions,
                draw_seed=seed + 30_000 + option_index * 1_000 + index,
                draws=draws,
                condition={"rescue": group.to_json(), "attenuation": strong},
            )
            recovered.append(bank.success - chosen_success[question.question_id])
        rescue_results.append(
            {"option": name, "group": group, "recovered": float(np.mean(recovered))}
        )
    best_rescue = max(rescue_results, key=lambda item: item["recovered"])
    rescue_recovery = float(best_rescue["recovered"] / max(1e-6, damage_value))

    # Matched controls.
    target_profile = _global_profile(profiles[best.group.layer])
    target_sorted = sorted(float(target_profile[c]) for c in best.group.channels)
    exclude = tuple(best.group.channels) + tuple(best_rescue["group"].channels)
    random_groups = []
    for index in range(config.random_replicates):
        channels = iv.matched_random_rank(
            target_profile,
            target_sorted,
            best.group.size,
            seed=seed + 40_000 + index,
            exclude=exclude,
        )
        random_groups.append(
            iv.ChannelGroup(layer=best.group.layer, channels=channels, label=f"random_{index}")
        )
    norm_matched = None
    try:
        norm_channels = iv.norm_matched_channels(
            target_profile, best.group.channels, best.group.size, exclude=exclude
        )
        norm_matched = iv.ChannelGroup(
            layer=best.group.layer, channels=norm_channels, label="norm_matched"
        )
    except ValueError:
        norm_matched = None

    other_group = None
    relations = sorted(profiles[best.group.layer])
    if len(relations) > 1:
        target_relation = max(relation_counts, key=relation_counts.get)
        other_relation = next(
            (relation for relation in relations if relation != target_relation), relations[0]
        )
        channels = iv.relation_specific_channels(
            profiles[best.group.layer], other_relation, best.group.size, exclude=exclude
        )
        other_group = iv.ChannelGroup(
            layer=best.group.layer, channels=channels, label="other_group", relation=other_relation
        )
    else:
        ranked = iv.top_contribution_channels(target_profile, 2 * best.group.size, exclude=exclude)
        other_group = iv.ChannelGroup(
            layer=best.group.layer,
            channels=tuple(sorted(ranked[-best.group.size :])),
            label="other_group",
        )

    engineering = _engineering_checks(env, screen_questions[0], best.group, strong)

    frozen = FrozenIntervention(
        target=best.group,
        mild_attenuation=mild,
        strong_attenuation=strong,
        rescue=best_rescue["group"],
        rescue_recovery_fraction=rescue_recovery,
        random_groups=random_groups,
        norm_matched_group=norm_matched,
        other_group=other_group,
        discovery={
            "sites": list(sites),
            "screened_grid": screened_grid,
            "chosen_attenuation": strong,
            "rescue_options": [
                {
                    "option": item["option"],
                    "group": item["group"].to_json(),
                    "recovered": item["recovered"],
                }
                for item in rescue_results
            ],
            "target_relation_counts": dict(relation_counts),
        },
    )
    return {
        "stage": "p2_discovery",
        "model": {"name": protocol.model_name, "revision": env.revision},
        "n_questions": len(questions),
        "screening_questions": len(screen_questions),
        "draws_per_condition": draws,
        "candidates": [candidate.to_json() for candidate in candidates],
        "refinements": refinements,
        "frozen": frozen.to_json(),
        "engineering_checks": engineering,
        "draws_used": int(
            len(screen_questions)
            * draws
            * (1 + len(candidates) + len(refinements) + len(rescue_options))
        ),
    }


def _engineering_checks(
    env: EpisodeEnvironment,
    question: Question,
    group: iv.ChannelGroup,
    attenuation: float,
) -> dict:
    """Direct undo and reinsertion must restore the prefixed run exactly."""

    ids, _ = env.prefix_ids(question)
    positions = _last_only(ids)
    with torch.no_grad():
        clean = run_prefix(env.model, ids, capture_hidden=False)
        attenuated = run_prefix(
            env.model,
            ids,
            hooks=iv.InterventionHooks(
                env.model, [(group, "attenuate", attenuation)], positions
            ),
            capture_hidden=False,
        )
        selection = iv.capture_selection(env.model, ids, group.layer, positions)
        removed_values = attenuation * selection[:, list(group.channels)]
        reinserted = run_prefix(
            env.model,
            ids,
            hooks=iv.InterventionHooks(
                env.model,
                [(group, "attenuate", attenuation), (group, "reinsert", removed_values)],
                positions,
            ),
            capture_hidden=False,
        )
    return {
        "attenuation_effective": bool(
            (clean.last_logits.float() - attenuated.last_logits.float()).abs().max().item() > 1e-4
        ),
        "reinsertion_max_abs_diff": float(
            (clean.last_logits.float() - reinserted.last_logits.float()).abs().max().item()
        ),
    }


# ---------------------------------------------------------------------------
# P2: locked validation
# ---------------------------------------------------------------------------
UNRELATED_QUESTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("What is 2 + 2?", ("4",)),
    ("What is the capital of France?", ("Paris",)),
    ("What color is the sky on a clear day?", ("blue",)),
    ("How many legs does a dog have?", ("4", "four")),
    ("What is the first month of the year?", ("January",)),
)


def run_p2_validation(
    env: EpisodeEnvironment,
    questions: list[Question],
    *,
    frozen: FrozenIntervention,
    protocol: Protocol = PROTOCOL,
    draws: int | None = None,
    random_replicates: int | None = None,
    seed: int = PROTOCOL.inner_seed,
    unrelated_check: bool = True,
) -> tuple[dict, dict[str, dict[str, OutcomeBank]]]:
    """Evaluate the frozen manipulation on independent questions."""

    draws = draws or protocol.sampling.eval_draws
    random_replicates = (
        random_replicates
        if random_replicates is not None
        else protocol.intervention.random_replicates
    )
    banks: dict[str, dict[str, OutcomeBank]] = {}
    for index, question in enumerate(questions):
        ids, _ = env.prefix_ids(question)
        positions = _last_only(ids)
        slots: dict[str, OutcomeBank] = {}
        slots["clean"] = bank_with_hooks(
            env,
            question,
            slot="clean",
            hooks_specs=[],
            positions=None,
            draw_seed=seed + index * 17,
            draws=draws,
            partition="validation",
        )
        for name, attenuation in (
            ("target_mild", frozen.mild_attenuation),
            ("target_strong", frozen.strong_attenuation),
        ):
            slots[name] = bank_with_hooks(
                env,
                question,
                slot=name,
                hooks_specs=[(frozen.target, "attenuate", attenuation)],
                positions=positions,
                draw_seed=seed + 1_000_000 + index * 17,
                draws=draws,
                partition="validation",
                condition={"group": frozen.target.to_json(), "attenuation": attenuation},
            )
        if frozen.rescue is not None:
            clean_values = iv.capture_group_activations(env.model, ids, frozen.rescue, positions)
            slots["restoration"] = bank_with_hooks(
                env,
                question,
                slot="restoration",
                hooks_specs=[
                    (frozen.target, "attenuate", frozen.strong_attenuation),
                    (frozen.rescue, "restore", clean_values),
                ],
                positions=positions,
                draw_seed=seed + 2_000_000 + index * 17,
                draws=draws,
                partition="validation",
                condition={"rescue": frozen.rescue.to_json()},
            )
        groups = list(frozen.random_groups[:random_replicates])
        if frozen.norm_matched_group is not None:
            groups.append(frozen.norm_matched_group)
        random_successes = []
        for group_index, group in enumerate(groups):
            bank = bank_with_hooks(
                env,
                question,
                slot=f"matched_random_{group_index}",
                hooks_specs=[(group, "attenuate", frozen.strong_attenuation)],
                positions=positions,
                draw_seed=seed + 3_000_000 + index * 17 + group_index,
                draws=draws,
                partition="validation",
                condition={"group": group.to_json(), "attenuation": frozen.strong_attenuation},
            )
            random_successes.append(bank.success)
            if group_index == 0:
                slots["matched_random"] = bank
        slots["matched_random"].condition["replicate_successes"] = random_successes
        if frozen.other_group is not None:
            slots["other_group"] = bank_with_hooks(
                env,
                question,
                slot="other_group",
                hooks_specs=[(frozen.other_group, "attenuate", frozen.strong_attenuation)],
                positions=positions,
                draw_seed=seed + 4_000_000 + index * 17,
                draws=draws,
                partition="validation",
                condition={"group": frozen.other_group.to_json()},
            )
        banks[question.question_id] = slots

    # Build per-question payloads with replicate successes first.
    items: list[ClusterItem] = []
    for question in questions:
        slots = banks[question.question_id]
        payload = {slot: bank.success for slot, bank in slots.items()}
        payload["matched_random_replicates"] = slots["matched_random"].condition[
            "replicate_successes"
        ]
        items.append(ClusterItem(subject_id=slots["clean"].subject_id, payload=payload))

    damage_values = [
        item.payload["clean"] - item.payload["target_strong"] for item in items
    ]
    damage = float(np.mean(damage_values))
    damage_ci = cluster_bootstrap(
        items,
        lambda index, all_items: float(
            np.mean(
                [
                    all_items[i].payload["clean"] - all_items[i].payload["target_strong"]
                    for i in index
                ]
            )
        ),
        seed=seed,
    )
    random_damage_values = [
        item.payload["clean"] - float(np.mean(item.payload["matched_random_replicates"]))
        for item in items
    ]
    random_damage = float(np.mean(random_damage_values))
    delta_random = damage - random_damage
    delta_random_ci = cluster_bootstrap(
        items,
        lambda index, all_items: float(
            np.mean(
                [
                    np.mean(all_items[i].payload["matched_random_replicates"])
                    - all_items[i].payload["target_strong"]
                    for i in index
                ]
            )
        ),
        seed=seed + 1,
    )
    restoration = None
    if "restoration" in banks[questions[0].question_id]:
        recovered_values = [
            item.payload["restoration"] - item.payload["target_strong"] for item in items
        ]
        recovered = float(np.mean(recovered_values))
        restoration = {
            "recovered": recovered,
            "recovery_fraction": recovered / damage if damage > 0 else None,
            "ci": cluster_bootstrap(
                items,
                lambda index, all_items: float(
                    np.mean(
                        [
                            all_items[i].payload["restoration"]
                            - all_items[i].payload["target_strong"]
                            for i in index
                        ]
                    )
                ),
                seed=seed + 2,
            ),
        }

    mild_damage = float(
        np.mean([item.payload["clean"] - item.payload["target_mild"] for item in items])
    )
    other_damage = (
        float(np.mean([item.payload["clean"] - item.payload["other_group"] for item in items]))
        if "other_group" in banks[questions[0].question_id]
        else None
    )
    per_relation: dict[str, dict] = {}
    for question, item in zip(questions, items):
        row = per_relation.setdefault(
            question.relation, {"n": 0, "damage": 0.0, "random_damage": 0.0}
        )
        row["n"] += 1
        row["damage"] += item.payload["clean"] - item.payload["target_strong"]
        row["random_damage"] += item.payload["clean"] - float(
            np.mean(item.payload["matched_random_replicates"])
        )
    for row in per_relation.values():
        row["damage"] /= max(1, row["n"])
        row["random_damage"] /= max(1, row["n"])

    gates = protocol.gates
    gate_damage = gate_decision(damage, damage_ci["ci_low"], gates.p2_target_damage)
    gate_random = gate_decision(
        delta_random, delta_random_ci["ci_low"], gates.p2_target_minus_random
    )
    gate_restoration = None
    if restoration is not None and restoration["recovery_fraction"] is not None:
        gate_restoration = {
            "value": restoration["recovery_fraction"],
            "threshold": gates.p2_restoration_recovery,
            "ci": restoration["ci"],
            "passed": restoration["recovery_fraction"] >= gates.p2_restoration_recovery,
        }
    unrelated = None
    if unrelated_check:
        unrelated = _unrelated_task_check(
            env, frozen, draws=max(2, draws // 4), seed=seed + 9
        )

    result = {
        "stage": "p2_validation",
        "model": {"name": protocol.model_name, "revision": env.revision},
        "n_questions": len(questions),
        "draws_per_condition": draws,
        "random_replicates": len(frozen.random_groups[:random_replicates]),
        "per_question": {
            question.question_id: {
                "subject_id": banks[question.question_id]["clean"].subject_id,
                "relation": question.relation,
                "slots": {
                    slot: bank.success
                    for slot, bank in banks[question.question_id].items()
                },
                "matched_random_replicates": banks[question.question_id][
                    "matched_random"
                ].condition["replicate_successes"],
            }
            for question in questions
        },
        "effects": {
            "damage_strong": damage,
            "damage_strong_ci": damage_ci,
            "damage_mild": mild_damage,
            "random_damage": random_damage,
            "target_minus_random": delta_random,
            "target_minus_random_ci": delta_random_ci,
            "other_group_damage": other_damage,
            "restoration": restoration,
            "per_relation": per_relation,
        },
        "gates": {
            "target_damage": gate_damage,
            "target_minus_random": gate_random,
            "restoration_recovery": gate_restoration,
        },
        "breakage_rates": bank_rates(banks),
        "unrelated_task": unrelated,
        "passed": bool(
            gate_damage["passed"]
            and gate_random["passed"]
            and (gate_restoration is None or gate_restoration["passed"])
        ),
    }
    return result, banks


def bank_rates(banks: dict[str, dict[str, OutcomeBank]]) -> dict:
    flat = []
    for slots in banks.values():
        for bank in slots.values():
            flat.append(bank.to_json())
    return audit_banks(flat)


def _unrelated_task_check(
    env: EpisodeEnvironment,
    frozen: FrozenIntervention,
    *,
    draws: int,
    seed: int,
) -> dict:
    """Generic breakage diagnostic on small unrelated questions."""

    damages = []
    truncations = []
    for index, (text, references) in enumerate(UNRELATED_QUESTIONS):
        question = Question(
            question_id=f"unrelated_{index}",
            subject_id=f"unrelated_{index}",
            subject_name="unrelated",
            relation="unrelated",
            question_text=text,
            references=references,
            source="unrelated_task",
        )
        ids, _ = env.prefix_ids(question)
        positions = _last_only(ids)
        clean = bank_with_hooks(
            env,
            question,
            slot="unrelated_clean",
            hooks_specs=[],
            positions=None,
            draw_seed=seed + index,
            draws=draws,
        )
        target = bank_with_hooks(
            env,
            question,
            slot="unrelated_target",
            hooks_specs=[(frozen.target, "attenuate", frozen.strong_attenuation)],
            positions=positions,
            draw_seed=seed + 100 + index,
            draws=draws,
        )
        damages.append(clean.success - target.success)
        truncations.extend(draw.truncated for draw in target.draws)
    return {
        "n_items": len(UNRELATED_QUESTIONS),
        "mean_damage": float(np.mean(damages)),
        "truncation_rate": float(np.mean(truncations)),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_p2_validation(result: dict, path: str | Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    effects = result["effects"]
    labels = ["mild", "strong", "restoration", "random", "other"]
    values = [
        effects["damage_mild"],
        effects["damage_strong"],
        effects["restoration"]["recovered"] if effects["restoration"] else float("nan"),
        effects["random_damage"],
        effects["other_group_damage"] if effects["other_group_damage"] is not None else float("nan"),
    ]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(
        labels,
        values,
        color=["tab:blue", "tab:blue", "tab:green", "tab:orange", "tab:red"],
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("success change vs clean")
    axis.set_title("P2 damage / control / restoration")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def plot_p2_per_question(banks: dict[str, dict[str, OutcomeBank]], path: str | Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [slots["clean"].success for slots in banks.values()]
    strong = [slots["target_strong"].success for slots in banks.values()]
    figure, axis = plt.subplots(figsize=(4.5, 4.5))
    axis.scatter(clean, strong, alpha=0.6, s=14)
    axis.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--")
    axis.set_xlabel("clean success")
    axis.set_ylabel("target strong success")
    axis.set_title("Per-question effect")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


# ---------------------------------------------------------------------------
# Later stages (gated)
# ---------------------------------------------------------------------------
def require_p2_pass(validation_path: str | Path) -> dict:
    payload = json.loads(Path(validation_path).read_text())
    if not payload.get("passed"):
        raise StageGated(
            "P2 did not pass; the intervention-dependent training branch stops here. "
            "Do not launch P3-P6 on a failed manipulation."
        )
    return payload


def run_p3(*args, **kwargs):  # pragma: no cover - gated
    raise StageGated("P3 requires the P2 gate to pass first; see plan section 10.")


def run_p4(*args, **kwargs):  # pragma: no cover - gated
    raise StageGated("P4 requires P3 results; see plan section 10.")


def run_p5(*args, **kwargs):  # pragma: no cover - gated
    raise StageGated("P5 requires a positive P3/P4 result; see plan section 10.")


def run_p6(*args, **kwargs):  # pragma: no cover - gated
    raise StageGated("P6 is an optional optimizer comparison; see plan section 10.")
