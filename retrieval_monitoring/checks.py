"""S1 apparatus checks on a real model.

The checks cover the plan's requirements:

- clean and intervened fork-versus-full-run equivalence, including a
  left-padded prefix with a preserved mask;
- correct subject-span location;
- a middle-layer intervention that reaches both branches, and a last-layer
  intervention that cannot;
- actual multi-token generation;
- clean replay from unmodified text.
"""

from __future__ import annotations

import torch

from .retained_state import (
    additive_transform,
    clean_replay,
    continue_branch,
    equivalence_tolerance,
    full_run,
    generate_branch,
    locate_span,
    run_prefix,
)


def _max_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max())


def run_state_check(
    loaded,
    *,
    subject_text: str = "Marie Curie",
    question: str = "In what city was Marie Curie born?",
    layer_index: int | None = None,
    max_new_tokens: int = 6,
) -> dict:
    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device
    dtype = next(model.parameters()).dtype
    tolerance = equivalence_tolerance(dtype)
    n_layers = int(model.config.num_hidden_layers)
    layer = layer_index if layer_index is not None else max(0, n_layers // 2 - 1)

    prefix_text = f"Subject: {subject_text}."
    suffix_text = f"\nQuestion: {question}\nAnswer:"
    prefix_ids = torch.tensor(
        [tokenizer(prefix_text, add_special_tokens=False)["input_ids"]], device=device
    )
    suffix_ids = torch.tensor(
        [tokenizer(suffix_text, add_special_tokens=False)["input_ids"]], device=device
    )
    span_start, span_end = locate_span(tokenizer, prefix_ids[0].tolist(), subject_text)

    checks: dict = {
        "prefix_len": int(prefix_ids.shape[1]),
        "suffix_len": int(suffix_ids.shape[1]),
        "subject_span": [span_start, span_end],
        "layer": layer,
        "n_layers": n_layers,
        "tolerance": tolerance,
    }

    # Clean fork versus full run (no padding).
    state = run_prefix(model, prefix_ids)
    branch = continue_branch(model, state, suffix_ids)
    reference = full_run(model, prefix_ids, suffix_ids)
    checks["clean_fork_vs_full"] = _max_diff(
        branch.logits, reference.logits[:, -suffix_ids.shape[1] :]
    )

    # Left-padded prefix with a preserved mask.
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    pad_width = 3
    padded_prefix = torch.full(
        (1, prefix_ids.shape[1] + pad_width), int(pad), dtype=torch.long, device=device
    )
    padded_prefix[:, pad_width:] = prefix_ids
    padded_mask = torch.zeros_like(padded_prefix)
    padded_mask[:, pad_width:] = 1
    padded_state = run_prefix(model, padded_prefix, attention_mask=padded_mask)
    padded_branch = continue_branch(model, padded_state, suffix_ids)
    padded_full_ids = torch.cat([padded_prefix, suffix_ids], dim=1)
    padded_full_mask = torch.cat([padded_mask, torch.ones_like(suffix_ids)], dim=1)
    padded_reference = full_run(model, padded_prefix, suffix_ids, attention_mask=padded_full_mask)
    checks["padded_fork_vs_full"] = _max_diff(
        padded_branch.logits, padded_reference.logits[:, -suffix_ids.shape[1] :]
    )

    # Intervention reaches both branches from a middle layer.
    vector = torch.zeros(model.config.hidden_size, device=device, dtype=dtype)
    vector[0] = 5.0
    intervened = run_prefix(
        model,
        prefix_ids,
        layer_index=layer,
        transform=additive_transform(vector, token_index=span_end - 1),
    )
    answer_intervened = continue_branch(model, intervened, suffix_ids)
    checks["middle_intervention_delta"] = _max_diff(answer_intervened.logits, branch.logits)

    last_layer = run_prefix(
        model,
        prefix_ids,
        layer_index=n_layers - 1,
        transform=additive_transform(vector, token_index=span_end - 1),
    )
    last_branch = continue_branch(model, last_layer, suffix_ids)
    checks["last_layer_delta"] = _max_diff(last_branch.logits, branch.logits)

    # Real multi-token generation and clean replay.
    generated = generate_branch(model, state, suffix_ids, max_new_tokens=max_new_tokens)
    checks["generated_tokens"] = int(generated.shape[1])
    replay = clean_replay(model, state, suffix_ids)
    checks["clean_replay_vs_full"] = _max_diff(
        replay.logits, reference.logits[:, -suffix_ids.shape[1] :]
    )

    checks["passed"] = bool(
        checks["clean_fork_vs_full"] < tolerance
        and checks["padded_fork_vs_full"] < tolerance
        and checks["middle_intervention_delta"] > 1e-3
        and checks["last_layer_delta"] < tolerance
        and checks["generated_tokens"] > 0
        and checks["clean_replay_vs_full"] < tolerance
    )
    return checks
