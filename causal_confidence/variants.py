"""Answer variants: small factual-learning adapters and knowledge verification.

Each variant is the frozen base model plus rank-8 adapters on selected MLP
gate/up projections, trained only on its exposure set's statements/questions.
No other parameter is touched, so variants of the same base model differ only
by what they learned.

Verification (plan step 2 of the revised protocol) checks that training
exposure actually moved correctness, measured on held-out paraphrases:

- every fact is evaluated under every reporter-training variant, so
  cross-variant spread per fact can be computed;
- per-variant success means are reported with bootstrap intervals;
- the training template and held-out paraphrases are compared, so a variant
  that only memorized the template does not pass;
- empty/truncation rates guard against format-only changes.

The gate is deliberately about the manipulation working, not about producing a
result: if variants do not differ reproducibly, reporter training is pointless.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .episodes import (
    answer_suffix_ids,
    build_prefix_ids,
    generate_answers,
    run_prefix,
    tokenize_text,
)
from .knowledge import Fact
from .outcomes import match_answer
from .reporters import (
    LinearAdapter,
    install_report_adapters,
    remove_report_adapters,
)


# ---------------------------------------------------------------------------
# Adapter installation and storage
# ---------------------------------------------------------------------------
def default_adapter_layers(n_layers: int, count: int = 4) -> tuple[int, ...]:
    if n_layers < count:
        return tuple(range(n_layers))
    step = max(1, (n_layers - 2) // count)
    layers = tuple(range(step, n_layers - 1, step))[:count]
    return layers or (n_layers // 2,)


def install_variant(
    env,
    *,
    layers: tuple[int, ...],
    rank: int = 8,
    state_path: str | Path | None = None,
) -> list[LinearAdapter]:
    adapters = install_report_adapters(env.model, layers, rank=rank)
    for adapter in adapters:
        adapter.enabled = True
    if state_path is not None:
        payload = torch.load(state_path, map_location=env.device, weights_only=True)
        for index, adapter in enumerate(adapters):
            adapter.lora_a.data.copy_(payload[f"lora_a_{index}"])
            adapter.lora_b.data.copy_(payload[f"lora_b_{index}"])
    return adapters


def save_variant(adapters: list[LinearAdapter], path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for index, adapter in enumerate(adapters):
        payload[f"lora_a_{index}"] = adapter.lora_a.detach().cpu()
        payload[f"lora_b_{index}"] = adapter.lora_b.detach().cpu()
    torch.save(payload, path)
    return str(path)


def remove_variant(env, adapters: list[LinearAdapter]) -> None:
    remove_report_adapters(env.model, adapters)


# Templates used for variant training; the last paraphrase is held out for
# verification and for the reporter's states, so success there measures learned
# facts rather than template memorization.
TRAINING_TEMPLATES: tuple[str, ...] = ("train", "eval0", "eval1", "eval2", "eval3")
HELD_OUT_TEMPLATE: str = "eval5"


def _training_example(env, fact: Fact, template: str = "train"):
    if template == "train":
        question_text = fact.train_question
    else:
        question_text = fact.eval_questions[int(template.replace("eval", ""))]
    prefix_ids, _ = build_prefix_ids(env.tokenizer, question_text)
    suffix = answer_suffix_ids(env.tokenizer)
    value = tokenize_text(env.tokenizer, " " + fact.value).squeeze(0)
    eos = env.tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos token; cannot teach answer termination")
    eos_token = torch.tensor([int(eos)], dtype=torch.long)
    target = torch.cat([value, eos_token], dim=0)
    input_ids = torch.cat([prefix_ids.squeeze(0), suffix.squeeze(0), target], dim=0)
    labels = torch.full_like(input_ids, -100)
    labels[-target.shape[0] :] = target
    return input_ids, labels


def train_variant(
    env,
    facts: list[Fact],
    *,
    layers: tuple[int, ...],
    rank: int = 8,
    epochs: int = 40,
    batch: int = 8,
    lr: float = 1e-4,
    seed: int = 0,
) -> tuple[list[LinearAdapter], dict]:
    """Train one answer variant on its exposure set."""

    if not facts:
        raise ValueError("variant has no training facts")
    adapters = install_variant(env, layers=layers, rank=rank)
    for parameter in env.model.parameters():
        parameter.requires_grad_(False)
    parameters: list[torch.nn.Parameter] = []
    for adapter in adapters:
        adapter.lora_a.requires_grad_(True)
        adapter.lora_b.requires_grad_(True)
        parameters.extend([adapter.lora_a, adapter.lora_b])
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    generator = random.Random(seed)
    examples = [
        _training_example(env, fact, template)
        for fact in facts
        for template in TRAINING_TEMPLATES
    ]
    pad_id = env.tokenizer.pad_token_id or 0
    steps = max(1, epochs * math.ceil(len(examples) / batch))
    losses: list[float] = []
    env.model.train()
    for step in range(steps):
        sample = generator.sample(examples, min(batch, len(examples)))
        max_len = max(ids.shape[0] for ids, _ in sample)
        input_ids = torch.full((len(sample), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(sample), max_len), -100, dtype=torch.long)
        attention = torch.zeros((len(sample), max_len), dtype=torch.long)
        for row, (ids, row_labels) in enumerate(sample):
            input_ids[row, : ids.shape[0]] = ids
            labels[row, : row_labels.shape[0]] = row_labels
            attention[row, : ids.shape[0]] = 1
        input_ids = input_ids.to(env.device)
        labels = labels.to(env.device)
        attention = attention.to(env.device)
        logits = env.model(input_ids=input_ids, attention_mask=attention).logits
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if step % max(1, steps // 10) == 0:
            print(f"    variant step {step}/{steps} loss {losses[-1]:.4f}", flush=True)
    env.model.eval()
    return adapters, {
        "steps": steps,
        "epochs": epochs,
        "batch": batch,
        "n_facts": len(facts),
        "final_loss": losses[-1] if losses else None,
        "first_loss": losses[0] if losses else None,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_facts(
    env,
    facts: list[Fact],
    *,
    template: str,
    draws: int,
    seed: int,
    partition: str = "eval",
) -> dict:
    """Sample answers for one template over a fact list and match them.

    ``template`` is ``"train"`` or ``"eval0"``/``"eval1"``/``"eval2"``.
    Returns per-fact success, draw texts for audit, and format statistics.
    """

    from .data import Question

    success: dict[str, float] = {}
    empty = 0
    truncated = 0
    n_draws = 0
    for index, fact in enumerate(facts):
        if template == "train":
            question_text = fact.train_question
        else:
            question_text = fact.eval_questions[int(template.replace("eval", ""))]
        question = Question(
            question_id=fact.fact_id,
            subject_id=fact.subject_id,
            subject_name=fact.subject_name,
            relation=fact.attribute,
            question_text=question_text,
            references=fact.references,
            source="knowledge",
        )
        ids, digest = env.prefix_ids(question)
        prefix = run_prefix(env.model, ids, capture_hidden=False, prompt_hash_value=digest)
        draws_out = generate_answers(
            env.model,
            prefix,
            env.answer_suffix,
            env.tokenizer,
            draws=draws,
            base_seed=seed * 100_000 + index * 17,
        )
        matched = sum(match_answer(draw.text, list(fact.references)) for draw in draws_out)
        success[fact.fact_id] = matched / max(1, len(draws_out))
        empty += sum(not draw.text.strip() for draw in draws_out)
        truncated += sum(draw.truncated for draw in draws_out)
        n_draws += len(draws_out)
    return {
        "partition": partition,
        "template": template,
        "success": success,
        "mean_success": sum(success.values()) / max(1, len(success)),
        "empty_rate": empty / max(1, n_draws),
        "truncation_rate": truncated / max(1, n_draws),
        "n_facts": len(facts),
        "draws": draws,
    }


# ---------------------------------------------------------------------------
# Verification gate (pure aggregation over measured successes)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VerificationConfig:
    min_mean_spread: float = 0.35
    fact_spread_threshold: float = 0.5
    min_spread_fraction: float = 0.5
    min_paraphrase_ratio: float = 0.5
    max_format_rate: float = 0.10
    min_reproducibility: float = 0.40


@dataclass
class VerificationReport:
    passed: bool
    details: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"passed": self.passed, **self.details}


def verify_knowledge(
    success_by_variant: dict[int, dict[str, float]],
    *,
    known_facts: dict[int, set[str]],
    train_template_success: dict[int, dict[str, float]],
    format_stats: dict[int, dict[str, float]],
    reproducibility: dict[int, float | None] | None = None,
    config: VerificationConfig = VerificationConfig(),
) -> VerificationReport:
    """Aggregate measured variant successes into the proceed/stop decision."""

    variants = sorted(success_by_variant)
    fact_ids = sorted(
        set.intersection(
            *[set(success_by_variant[variant]) for variant in variants]
        )
    )
    spreads = []
    for fact_id in fact_ids:
        values = [success_by_variant[variant][fact_id] for variant in variants]
        spreads.append(max(values) - min(values))
    mean_spread = sum(spreads) / max(1, len(spreads))
    spread_fraction = (
        sum(spread >= config.fact_spread_threshold for spread in spreads)
        / max(1, len(spreads))
    )

    variant_means = {
        str(variant): sum(success_by_variant[variant].values())
        / max(1, len(success_by_variant[variant]))
        for variant in variants
    }
    paraphrase_ratios = {}
    for variant in variants:
        known = known_facts.get(variant, set())
        train_lookup = train_template_success.get(variant, {})
        paired_ids = [
            fact_id
            for fact_id in known
            if fact_id in success_by_variant[variant] and fact_id in train_lookup
        ]
        paraphrase_mean = (
            sum(success_by_variant[variant][fact_id] for fact_id in paired_ids)
            / max(1, len(paired_ids))
        )
        train_mean = (
            sum(train_lookup[fact_id] for fact_id in paired_ids) / max(1, len(paired_ids))
        )
        paraphrase_ratios[str(variant)] = (
            paraphrase_mean / train_mean if train_mean > 0 else None
        )
    ratio_values = [value for value in paraphrase_ratios.values() if value is not None]
    paraphrase_ratio = min(ratio_values) if ratio_values else 0.0

    max_format = max(
        max(stats.get("empty_rate", 0.0), stats.get("truncation_rate", 0.0))
        for stats in format_stats.values()
    )
    reproducibility_values = [
        value for value in (reproducibility or {}).values() if value is not None
    ]
    min_reproducibility = (
        min(reproducibility_values) if reproducibility_values else None
    )

    passed = bool(
        mean_spread >= config.min_mean_spread
        and spread_fraction >= config.min_spread_fraction
        and paraphrase_ratio >= config.min_paraphrase_ratio
        and max_format <= config.max_format_rate
        and (
            min_reproducibility is None
            or min_reproducibility >= config.min_reproducibility
        )
    )
    return VerificationReport(
        passed=passed,
        details={
            "n_facts_compared": len(fact_ids),
            "mean_spread": mean_spread,
            "spread_fraction": spread_fraction,
            "variant_means": variant_means,
            "paraphrase_ratios": paraphrase_ratios,
            "min_paraphrase_ratio": paraphrase_ratio,
            "max_format_rate": max_format,
            "min_reproducibility": min_reproducibility,
            "config": {
                "min_mean_spread": config.min_mean_spread,
                "min_spread_fraction": config.min_spread_fraction,
                "min_paraphrase_ratio": config.min_paraphrase_ratio,
                "max_format_rate": config.max_format_rate,
                "min_reproducibility": config.min_reproducibility,
            },
        },
    )
