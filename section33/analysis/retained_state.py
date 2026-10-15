"""Retained-state protocol for the retrieval-monitoring plan.

The Section 3.3 collector asks the confidence question in a fresh forward pass
(``confidence_prompt_ids``).  A hook that changes the answer pass therefore does
not reach the confidence report.  This module implements the fix: run the
prefix once, keep the KV cache, and fork the same internal episode into an
answer branch and a confidence branch.

Contract
--------
1. The prefix runs once per intervention condition.  Any hook modifies the
   prefix state before the fork point.
2. Every branch continues from a deep copy of the retained cache.  A branch
   cannot change the retained state or another branch.
3. Branch outputs equal the outputs of an independent full run.  The unit
   tests check this exactly on a tiny model.

The module only handles state retention and branching.  Prompt construction,
interventions, and scoring stay in the caller.

Intervention site rule
----------------------
A continuation reads cached keys and values.  It does not read the fork-point
hidden state.  An intervention must therefore run at a decoder layer whose
later layers recompute keys and values from the changed state.  The last
decoder layer does not satisfy this rule.  Use a middle layer for subject-state
interventions.
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
    prefix_len: int


def additive_transform(vector: torch.Tensor, token_index: int = -1) -> Callable:
    """Return a hook transform that adds ``vector`` to one prefix token.

    The transform receives the decoder-layer output ``(batch, seq, dim)`` and
    returns a modified copy.  ``token_index=-1`` means the last prefix token,
    which is the usual intervention site.  The vector must broadcast to the
    hidden width.
    """

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.clone()
        hidden[:, token_index, :] = hidden[:, token_index, :] + vector
        return hidden

    return transform


def _apply_hook(model, layer_index: int, transform: Callable):
    """Attach a forward hook that rewrites one decoder-layer output."""

    def hook(module, args, output):
        hidden = output if torch.is_tensor(output) else output[0]
        return transform(hidden)

    return model.model.layers[layer_index].register_forward_hook(hook)


def run_prefix(
    model,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    layer_index: int | None = None,
    transform: Callable | None = None,
    use_cache: bool = True,
) -> PrefixState:
    """Run the prefix once and retain the state at the fork point.

    ``layer_index`` and ``transform`` select an optional intervention.  The
    hook runs inside the prefix pass, so its effect is inside the retained
    cache and reaches every later branch.
    """

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    handle = None
    if layer_index is not None and transform is not None:
        handle = _apply_hook(model, layer_index, transform)
    try:
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                output_hidden_states=True,
            )
    finally:
        if handle is not None:
            handle.remove()
    if not use_cache or output.past_key_values is None:
        raise ValueError("run_prefix needs use_cache=True")
    return PrefixState(
        cache=output.past_key_values,
        last_hidden=output.hidden_states[-1].detach().clone(),
        input_ids=input_ids.detach().clone(),
        attention_mask=attention_mask.detach().clone(),
        prefix_len=int(input_ids.shape[1]),
    )


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
    retained state cannot change.  The suffix must not include the prefix
    tokens.
    """

    cache = copy.deepcopy(state.cache) if fork else state.cache
    suffix_len = int(suffix_ids.shape[1])
    total_len = state.prefix_len + suffix_len
    attention_mask = torch.ones(
        (suffix_ids.shape[0], total_len), dtype=torch.long, device=suffix_ids.device
    )
    cache_position = torch.arange(state.prefix_len, total_len, device=suffix_ids.device)
    with torch.no_grad():
        output = model(
            input_ids=suffix_ids,
            attention_mask=attention_mask,
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
    """Continue several named branches from the same retained state.

    Each branch gets its own deep copy when ``fork`` is true.  The returned
    dict preserves the branch order of the input.
    """

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
