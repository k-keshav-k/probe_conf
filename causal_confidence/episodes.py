"""Prefix interventions, retained-state branching, adapter gating, and checks.

Contract (plan section 5):

1. The manipulation applies only at declared prefix positions.  Hooks run
   during the prefix pass and are removed before any continuation, so no hook
   ever re-fires on generated or suffix tokens.
2. A middle-layer intervention propagates into later cached keys and values.
   The prefix's final logits are kept so a first answer token can be sampled
   from the intervened state.
3. Every branch gets a deep copy of the retained cache.  Generating confidence
   cannot mutate later answer samples.
4. A cached branch equals a complete recomputation with the same condition and
   suffix, including the first-token logits, attention masks, and positions.
5. Reporter adapters are disabled for answer generation.  The shared prefix is
   detached and every answer-policy parameter is frozen during reporting.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .interventions import InterventionHooks
from .protocol import ANSWER_SUFFIX, PREFIX_TEMPLATE, REPORT_SUFFIX, SYSTEM_PROMPT


@dataclass
class PrefixRun:
    """Retained internal episode at the fork point."""

    cache: object
    last_hidden: torch.Tensor
    last_logits: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prefix_len: int
    prompt_hash: str = ""
    perturbation_norms: dict = field(default_factory=dict)


@dataclass
class Draw:
    """One sampled answer (or one greedy report) result."""

    text: str
    token_ids: list[int]
    n_tokens: int
    finished: bool
    truncated: bool
    stop_reason: str
    logprob: float | None = None


def tokenize_text(tokenizer, text: str) -> torch.Tensor:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return encoded.input_ids


def prompt_hash(tokenizer, text: str) -> str:
    name = getattr(tokenizer, "name_or_path", "tokenizer")
    return hashlib.sha256(f"{name}||{text}".encode()).hexdigest()[:16]


def build_prefix_text(question: str, *, system_prompt: str = SYSTEM_PROMPT) -> str:
    return PREFIX_TEMPLATE.format(system=system_prompt, question=question)


def build_prefix_ids(tokenizer, question: str, *, system_prompt: str = SYSTEM_PROMPT):
    text = build_prefix_text(question, system_prompt=system_prompt)
    return tokenize_text(tokenizer, text), prompt_hash(tokenizer, text)


def answer_suffix_ids(tokenizer, suffix: str = ANSWER_SUFFIX) -> torch.Tensor:
    return tokenize_text(tokenizer, suffix)


def report_suffix_ids(tokenizer, suffix: str = REPORT_SUFFIX) -> torch.Tensor:
    return tokenize_text(tokenizer, suffix)


# ---------------------------------------------------------------------------
# Prefix run
# ---------------------------------------------------------------------------
def run_prefix(
    model,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    hooks: InterventionHooks | None = None,
    capture_hidden: bool = True,
    prompt_hash_value: str = "",
) -> PrefixRun:
    """Run the prefix once and retain the state at the fork point.

    ``hooks`` (an :class:`InterventionHooks`) apply inside this pass only.
    """

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    entered = False
    if hooks is not None and not hooks.active:
        hooks.__enter__()
        entered = True
    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=capture_hidden,
            )
    finally:
        if hooks is not None and entered:
            hooks.__exit__(None, None, None)
    if output.past_key_values is None:
        raise ValueError("run_prefix needs use_cache=True")
    hidden = (
        output.hidden_states[-1].detach().clone() if capture_hidden else torch.empty(0)
    )
    norms = hooks.perturbation_summary() if hooks is not None else {}
    return PrefixRun(
        cache=output.past_key_values,
        last_hidden=hidden,
        last_logits=output.logits[:, -1, :].detach().clone(),
        input_ids=input_ids.detach().clone(),
        attention_mask=attention_mask.detach().clone(),
        prefix_len=int(input_ids.shape[1]),
        prompt_hash=prompt_hash_value,
        perturbation_norms=norms,
    )


def _suffix_forward(
    model, cache: object, prefix: PrefixRun, suffix_ids: torch.Tensor, *, requires_grad: bool = False
):
    """First forward over a suffix against the retained cache.

    ``requires_grad`` is False for every diagnostic path; reporter training
    turns it on so suffix activations carry gradients while the shared prefix
    stays detached.
    """

    total_len = prefix.prefix_len + suffix_ids.shape[1]
    attention_mask = torch.cat(
        [
            prefix.attention_mask,
            torch.ones(
                (prefix.attention_mask.shape[0], suffix_ids.shape[1]),
                dtype=prefix.attention_mask.dtype,
                device=prefix.attention_mask.device,
            ),
        ],
        dim=1,
    )

    def run():
        return model(
            input_ids=suffix_ids,
            attention_mask=attention_mask,
            past_key_values=cache,
            cache_position=torch.arange(
                prefix.prefix_len, total_len, device=suffix_ids.device
            ),
            use_cache=True,
        )

    if requires_grad:
        return run()
    with torch.no_grad():
        return run()


def continue_teacher_forced(
    model,
    prefix: PrefixRun,
    suffix_ids: torch.Tensor,
    *,
    fork: bool = True,
    requires_grad: bool = False,
):
    """Teacher-forced logits at every suffix position (does not mutate prefix)."""

    cache = copy.deepcopy(prefix.cache) if fork else prefix.cache
    output = _suffix_forward(
        model, cache, prefix, suffix_ids, requires_grad=requires_grad
    )
    return output.logits


def _step_token(model, cache: object, token: int, total_len: int, device):
    step = torch.tensor([[token]], device=device)
    attention_mask = torch.ones((1, total_len + 1), dtype=torch.long, device=device)
    return model(
        input_ids=step,
        attention_mask=attention_mask,
        past_key_values=cache,
        cache_position=torch.arange(total_len, total_len + 1, device=device),
        use_cache=True,
    )


def continue_greedy(
    model,
    prefix: PrefixRun,
    suffix_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    stop_token_ids: tuple[int, ...] = (),
) -> Draw:
    """Greedy continuation (used by the constrained numeric report decoder)."""

    cache = copy.deepcopy(prefix.cache)
    output = _suffix_forward(model, cache, prefix, suffix_ids)
    cache = output.past_key_values
    total_len = prefix.prefix_len + suffix_ids.shape[1]
    next_logits = output.logits[0, -1]
    generated: list[int] = []
    finished = False
    stop_reason = "max_new_tokens"
    while len(generated) < max_new_tokens:
        token = int(torch.argmax(next_logits.float()).item())
        if token in stop_token_ids:
            finished = True
            stop_reason = "eos"
            break
        generated.append(token)
        output = _step_token(model, cache, token, total_len, suffix_ids.device)
        cache = output.past_key_values
        total_len += 1
        next_logits = output.logits[0, -1]
    return Draw(
        text="",
        token_ids=generated,
        n_tokens=len(generated),
        finished=finished,
        truncated=not finished,
        stop_reason=stop_reason,
    )


def _sample_token(
    logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator
) -> tuple[int, float]:
    scores = logits.float() / max(float(temperature), 1e-6)
    if top_p < 1.0:
        sorted_scores, sorted_index = torch.sort(scores, descending=True)
        probabilities = F.softmax(sorted_scores, dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        cutoff = int(torch.searchsorted(cumulative, top_p, right=True).item()) + 1
        kept = sorted_index[:cutoff]
        mask = torch.full_like(scores, float("-inf"))
        mask[kept] = scores[kept]
        scores = mask
    probabilities = F.softmax(scores, dim=-1)
    token = int(torch.multinomial(probabilities, 1, generator=generator).item())
    logprob = float(F.log_softmax(scores, dim=-1)[token].item())
    return token, logprob


def generate_answer(
    model,
    prefix: PrefixRun,
    suffix_ids: torch.Tensor,
    tokenizer,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 32,
    seed: int = 0,
    stop_strings: tuple[str, ...] = ("\n",),
) -> Draw:
    """Sample one answer draw from the retained state.

    The first new token is sampled from the suffix's final logits inside the
    retained state; each later token is sampled from the cached continuation.
    The loop is manual so each draw has an independent seed and a private
    cache copy.
    """

    cache = copy.deepcopy(prefix.cache)
    output = _suffix_forward(model, cache, prefix, suffix_ids)
    cache = output.past_key_values
    total_len = prefix.prefix_len + suffix_ids.shape[1]
    next_logits = output.logits[0, -1]
    generator = torch.Generator(device=next_logits.device)
    generator.manual_seed(seed)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    generated: list[int] = []
    logprob_sum = 0.0
    finished = False
    stop_reason = "max_new_tokens"
    while len(generated) < max_new_tokens:
        token, logprob = _sample_token(next_logits, temperature, top_p, generator)
        logprob_sum += logprob
        if eos_id is not None and token == eos_id:
            finished = True
            stop_reason = "eos"
            break
        generated.append(token)
        output = _step_token(model, cache, token, total_len, suffix_ids.device)
        cache = output.past_key_values
        total_len += 1
        next_logits = output.logits[0, -1]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if any(stop in text for stop in stop_strings):
            finished = True
            stop_reason = "stop_string"
            break
    text = tokenizer.decode(generated, skip_special_tokens=True)
    for stop in stop_strings:
        if stop in text:
            text = text.split(stop)[0]
    return Draw(
        text=text.strip(),
        token_ids=generated,
        n_tokens=len(generated),
        finished=finished,
        truncated=not finished,
        stop_reason=stop_reason,
        logprob=logprob_sum,
    )


def generate_answers(
    model,
    prefix: PrefixRun,
    suffix_ids: torch.Tensor,
    tokenizer,
    *,
    draws: int,
    base_seed: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 32,
    stop_strings: tuple[str, ...] = ("\n",),
) -> list[Draw]:
    """Independent draws from the frozen policy for one retained state."""

    return [
        generate_answer(
            model,
            prefix,
            suffix_ids,
            tokenizer,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=base_seed + index,
            stop_strings=stop_strings,
        )
        for index in range(draws)
    ]


# ---------------------------------------------------------------------------
# Adapter gating
# ---------------------------------------------------------------------------
class adapters_enabled:
    """Context manager enabling or disabling reporter adapters."""

    def __init__(self, adapters, enabled: bool):
        self.adapters = list(adapters)
        self.enabled = enabled
        self.previous: list[bool] = []

    def __enter__(self):
        self.previous = [adapter.enabled for adapter in self.adapters]
        for adapter in self.adapters:
            adapter.enabled = self.enabled
        return self

    def __exit__(self, exc_type, exc, tb):
        for adapter, value in zip(self.adapters, self.previous):
            adapter.enabled = value
        return None


# ---------------------------------------------------------------------------
# Contract checks
# ---------------------------------------------------------------------------
def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def duplicate_run_tolerance(model, input_ids: torch.Tensor, *, runs: int = 3) -> dict:
    """Numerical tolerance from duplicate no-hook prefix runs."""

    outputs = [run_prefix(model, input_ids, capture_hidden=False) for _ in range(runs)]
    logits = [item.last_logits for item in outputs]
    differences = [max_abs_diff(logits[0], logits[index]) for index in range(1, runs)]
    return {
        "runs": runs,
        "max_logit_difference": max(differences, default=0.0),
        "tolerance": max(max(differences, default=0.0), 1e-4),
    }


def cache_full_run_equivalence(
    model,
    input_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    *,
    hook_specs,
    positions,
    attention_mask: torch.Tensor | None = None,
) -> dict:
    """Compare a cached branch to a full recomputation with the same condition."""

    with InterventionHooks(model, hook_specs, positions) as hooks:
        prefix = run_prefix(
            model, input_ids, attention_mask=attention_mask, hooks=hooks, capture_hidden=False
        )
    cached_logits = continue_teacher_forced(model, prefix, suffix_ids)

    full_ids = torch.cat([input_ids, suffix_ids], dim=1)
    full_mask = (
        torch.ones_like(full_ids)
        if attention_mask is None
        else torch.cat([attention_mask, torch.ones_like(suffix_ids)], dim=1)
    )
    with InterventionHooks(model, hook_specs, positions) as hooks:
        with torch.no_grad():
            full = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
    full_logits = full.logits[:, input_ids.shape[1] :, :]
    return {
        "max_abs_diff": max_abs_diff(cached_logits, full_logits),
        "n_suffix_tokens": int(suffix_ids.shape[1]),
    }


def last_layer_cache_miss(
    model,
    input_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    *,
    layer_index: int,
    positions,
    attenuation: float = 0.5,
    channels: tuple[int, ...] | None = None,
) -> dict:
    """Diagnostic: a last-layer hook cannot reach the continuation cache."""

    from .interventions import ChannelGroup

    if channels is None:
        intermediate = model.config.intermediate_size
        channels = tuple(range(min(16, intermediate)))
    group = ChannelGroup(layer=layer_index, channels=channels, label="last_layer_probe")
    result = cache_full_run_equivalence(
        model,
        input_ids,
        suffix_ids,
        hook_specs=[(group, "attenuate", attenuation)],
        positions=positions,
    )
    result["layer_index"] = layer_index
    result["cache_reaches"] = result["max_abs_diff"] > 1e-3
    return result


def check_prefix_immutability(model, prefix: PrefixRun, suffix_ids: torch.Tensor) -> dict:
    """Deep-copied branches must not mutate the retained cache or each other."""

    before_len = prefix.cache.get_seq_length()
    before_logits = prefix.last_logits.clone()
    first = continue_teacher_forced(model, prefix, suffix_ids)
    second = continue_teacher_forced(model, prefix, suffix_ids)
    after_len = prefix.cache.get_seq_length()
    return {
        "cache_length_before": int(before_len),
        "cache_length_after": int(after_len),
        "cache_unchanged": before_len == after_len,
        "branch_repeat_max_abs_diff": max_abs_diff(first, second),
        "last_logits_unchanged": max_abs_diff(before_logits, prefix.last_logits) == 0.0,
    }


def check_adapter_gating(model, prefix: PrefixRun, suffix_ids: torch.Tensor, adapters) -> dict:
    """Reporter adapters must not change answer-branch logits when disabled."""

    with adapters_enabled(adapters, False):
        disabled = continue_teacher_forced(model, prefix, suffix_ids)
    with adapters_enabled(adapters, True):
        enabled = continue_teacher_forced(model, prefix, suffix_ids)
    with adapters_enabled(adapters, False):
        disabled_again = continue_teacher_forced(model, prefix, suffix_ids)
    return {
        "disabled_vs_disabled_max_abs_diff": max_abs_diff(disabled, disabled_again),
        "enabled_vs_disabled_max_abs_diff": max_abs_diff(enabled, disabled),
        "answer_branch_stable": max_abs_diff(disabled, disabled_again) == 0.0,
    }


def frozen_parameter_check(model, trainable_patterns: tuple[str, ...], step_fn) -> dict:
    """Run one optimizer step and verify only declared parameters changed.

    ``step_fn`` performs the forward/backward/step.  ``trainable_patterns``
    are substring matches against parameter names (for example ``("lora_",)``).
    """

    tracked = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    step_fn(model)
    changed_trainable = [
        name
        for name, before in tracked.items()
        if not torch.equal(before, model.get_parameter(name).detach())
    ]
    changed_frozen = [
        name
        for name, before in frozen.items()
        if not torch.equal(before, model.get_parameter(name).detach())
    ]
    unexpected = [
        name
        for name in changed_trainable
        if not any(pattern in name for pattern in trainable_patterns)
    ]
    return {
        "n_trainable": len(tracked),
        "n_frozen": len(frozen),
        "changed_trainable": changed_trainable,
        "changed_frozen": changed_frozen,
        "unexpected_trainable_changes": unexpected,
        "passed": not changed_frozen and not unexpected,
    }
