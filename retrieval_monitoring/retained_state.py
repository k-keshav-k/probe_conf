"""Retained-state protocol with correct attention-mask handling.

The plan requires one internal episode per intervention condition:

1. Run a fixed prefix once.  Any intervention modifies the prefix state before
   the fork point.
2. Fork the retained KV cache into an answer branch and a confidence branch.
   A branch cannot change the retained state or another branch.
3. Each branch equals an independent full run of the same text under the same
   mask, within a declared numerical tolerance.

The earlier version replaced the prefix attention mask with all ones on
continuation.  A padded-prefix check differed from the correctly masked run by
0.0351 in maximum logit value.  This module preserves the prefix mask, derives
position ids from the mask, and rejects masks that are not a single contiguous
block (left or right padding only).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class PrefixState:
    """Retained internal episode at the fork point."""

    cache: object
    last_hidden: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    prefix_len: int


def equivalence_tolerance(dtype: torch.dtype) -> float:
    """Numerical tolerance for branch-versus-full-run equivalence checks."""

    if dtype in (torch.float16, torch.bfloat16):
        return 0.1
    return 1e-3


def positions_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Absolute positions of the attended tokens, zeros for padding."""

    return (attention_mask.cumsum(dim=-1) - 1).clamp_min(0)


def _validate_mask(attention_mask: torch.Tensor) -> None:
    """Accept only left- or right-padded masks with at least one real token."""

    for row in attention_mask:
        values = [int(value) for value in row.tolist()]
        transitions = sum(1 for left, right in zip(values, values[1:]) if left != right)
        if transitions > 1:
            raise ValueError(
                "attention mask must be left- or right-padded (one contiguous block)"
            )
        if not any(values):
            raise ValueError("attention mask row has no attended token")


def _apply_hook(model, layer_index: int, transform: Callable):
    """Attach a forward hook that rewrites one decoder-layer output."""

    def hook(module, args, output):
        hidden = output if torch.is_tensor(output) else output[0]
        return transform(hidden)

    return model.model.layers[layer_index].register_forward_hook(hook)


def additive_transform(vector: torch.Tensor, token_index: int = -1) -> Callable:
    """Return a hook transform that adds ``vector`` to one prefix token."""

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.clone()
        hidden[:, token_index, :] = hidden[:, token_index, :] + vector
        return hidden

    return transform


def run_prefix(
    model,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    layer_index: int | None = None,
    transform: Callable | None = None,
) -> PrefixState:
    """Run the prefix once and retain the state at the fork point."""

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    _validate_mask(attention_mask)
    position_ids = positions_from_mask(attention_mask)
    handle = None
    if layer_index is not None and transform is not None:
        handle = _apply_hook(model, layer_index, transform)
    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                output_hidden_states=True,
            )
    finally:
        if handle is not None:
            handle.remove()
    if output.past_key_values is None:
        raise ValueError("run_prefix needs a model that returns past_key_values")
    return PrefixState(
        cache=output.past_key_values,
        last_hidden=output.hidden_states[-1].detach().clone(),
        input_ids=input_ids.detach().clone(),
        attention_mask=attention_mask.detach().clone(),
        position_ids=position_ids.detach().clone(),
        prefix_len=int(input_ids.shape[1]),
    )


def _total_mask_and_positions(state: PrefixState, suffix_ids: torch.Tensor):
    ones = torch.ones(
        (suffix_ids.shape[0], suffix_ids.shape[1]),
        dtype=state.attention_mask.dtype,
        device=suffix_ids.device,
    )
    total_mask = torch.cat([state.attention_mask.to(suffix_ids.device), ones], dim=1)
    total_positions = positions_from_mask(total_mask)
    return total_mask, total_positions[:, state.prefix_len :]


def continue_branch(
    model,
    state: PrefixState,
    suffix_ids: torch.Tensor,
    *,
    fork: bool = True,
    output_hidden_states: bool = False,
):
    """Continue one branch from the retained state.

    ``fork=True`` gives the branch a private deep copy of the cache.  The
    suffix must not include the prefix tokens.  The prefix mask is preserved
    and the suffix mask is appended.
    """

    cache = copy.deepcopy(state.cache) if fork else state.cache
    suffix_len = int(suffix_ids.shape[1])
    total_mask, suffix_positions = _total_mask_and_positions(state, suffix_ids)
    cache_position = torch.arange(
        state.prefix_len, state.prefix_len + suffix_len, device=suffix_ids.device
    )
    with torch.no_grad():
        output = model(
            input_ids=suffix_ids,
            attention_mask=total_mask,
            position_ids=suffix_positions,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
    return output


def fork_branches(
    model,
    state: PrefixState,
    branches: dict[str, torch.Tensor],
    *,
    fork: bool = True,
    output_hidden_states: bool = False,
) -> dict:
    """Continue several named branches from the same retained state."""

    return {
        name: continue_branch(
            model,
            state,
            suffix,
            fork=fork,
            output_hidden_states=output_hidden_states,
        )
        for name, suffix in branches.items()
    }


@torch.no_grad()
def generate_branch(
    model,
    state: PrefixState,
    suffix_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    do_sample: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    seed: int | None = None,
    stop_token_ids: tuple[int, ...] = (),
) -> torch.Tensor:
    """Generate real tokens for one branch from the retained state.

    Returns the new token ids only, with shape ``(batch, n_generated)``.
    Stop tokens are removed from the result.
    """

    cache = copy.deepcopy(state.cache)
    total_mask, _ = _total_mask_and_positions(state, suffix_ids)
    if seed is not None:
        torch.manual_seed(seed)
    kwargs = dict(
        input_ids=suffix_ids,
        attention_mask=total_mask,
        past_key_values=cache,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        use_cache=True,
    )
    if do_sample:
        kwargs["temperature"] = temperature if temperature is not None else 1.0
        kwargs["top_p"] = top_p if top_p is not None else 1.0
    output = model.generate(**kwargs)
    generated = output[:, suffix_ids.shape[1] :]
    if stop_token_ids:
        stop = set(stop_token_ids)
        trimmed = []
        for row in generated:
            ids = [int(token) for token in row.tolist()]
            for index, token in enumerate(ids):
                if token in stop:
                    ids = ids[:index]
                    break
            trimmed.append(ids)
        width = max((len(ids) for ids in trimmed), default=0)
        padded = torch.full(
            (len(trimmed), width), 0, dtype=generated.dtype, device=generated.device
        )
        for row_index, ids in enumerate(trimmed):
            if ids:
                padded[row_index, : len(ids)] = torch.tensor(
                    ids, dtype=generated.dtype, device=generated.device
                )
        generated = padded
    return generated


def full_run(
    model,
    prefix_ids: torch.Tensor,
    suffix_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
):
    """Independent full run of the same text, with no cache."""

    ids = torch.cat([prefix_ids, suffix_ids], dim=1)
    if attention_mask is None:
        attention_mask = torch.ones_like(ids)
    _validate_mask(attention_mask)
    with torch.no_grad():
        return model(
            input_ids=ids,
            attention_mask=attention_mask,
            position_ids=positions_from_mask(attention_mask),
            use_cache=False,
        )


def locate_span(
    tokenizer,
    ids: list[int],
    text: str,
    prefix_text: str | None = None,
) -> tuple[int, int]:
    """Find the token span of ``text`` inside ``ids`` (first occurrence).

    First tries exact token-subsequence matches, with and without a leading
    space and with the first token dropped.  If that fails, ``prefix_text``
    enables a character-offset fallback, which handles punctuation merges such
    as ``"Jr."`` inside ``"Subject: Friedrich Ebert Jr.."``.  Raises
    ValueError when no variant matches.
    """

    candidates: list[list[int]] = []
    truncated: list[list[int]] = []
    for variant in (text, " " + text, text.strip()):
        needle = tokenizer(variant, add_special_tokens=False)["input_ids"]
        if not needle:
            continue
        candidates.append(list(needle))
        if len(needle) > 1:
            truncated.append(list(needle[1:]))
    for needle in candidates + truncated:
        for start in range(0, len(ids) - len(needle) + 1):
            if ids[start : start + len(needle)] == needle:
                return start, start + len(needle)

    if prefix_text is not None and text in prefix_text:
        try:
            encoded = tokenizer(
                prefix_text, add_special_tokens=False, return_offsets_mapping=True
            )
        except (TypeError, ValueError):
            encoded = None
        if encoded is not None and list(encoded["input_ids"]) == list(ids):
            start_char = prefix_text.index(text)
            end_char = start_char + len(text)
            span = [
                index
                for index, (start_offset, end_offset) in enumerate(encoded["offset_mapping"])
                if end_offset > start_char and start_offset < end_char
            ]
            if span:
                return span[0], span[-1] + 1

    raise ValueError(f"span not found in token ids: {text!r}")


def clean_replay(
    model,
    state: PrefixState,
    suffix_ids: torch.Tensor,
):
    """Rebuild the same episode from unmodified text (no cache, no hook)."""

    return full_run(model, state.input_ids, suffix_ids)
