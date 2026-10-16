"""Native confidence training arms A/B/C.

The reporter emits an actual number from the fixed grid ``0.00 ... 1.00``.
Training optimizes the proper score directly:

    L = mean_k  sum_g p(g) * (g - Y_k)^2   (+ beta * KL(answer || frozen answer))

where ``p`` is the report distribution over grid strings, computed by walking
a trie over their tokenizations against the retained prefix cache, and
``Y_k`` are independently graded answers from a frozen snapshot of the
knowledge variant.

Arms (identical examples and budget):

- **A — native reporting**: the reporter adapter is active only while scoring
  the report suffix; the question prefix is frozen.
- **B — shared computation**: the reporter adapter is active while processing
  the question and the report, so training can change the representation the
  answer itself is computed from.  An answer-preservation term keeps the
  adapted answer distribution close to the frozen one.
- **C — difficulty control**: same architecture and data as B, but the target
  is success averaged across variants of the same question (question-level
  difficulty), not the variant's own outcomes.

Emitted reports are evaluated as actually decoded numbers (greedy constrained
decoding), not external means.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .episodes import adapters_enabled, tokenize_text
from .protocol import CONFIDENCE_GRID
from .reporters import reporter_parameters


# ---------------------------------------------------------------------------
# Grid grammar
# ---------------------------------------------------------------------------
@dataclass
class GridGrammar:
    strings: tuple[str, ...]
    values: torch.Tensor
    sequences: list[list[int]]
    trie: dict
    max_depth: int

    def children(self, node: dict) -> list[int]:
        return [token for token in node if token != "__terminal__"]

    def terminal(self, node: dict) -> int | None:
        return node.get("__terminal__")


def build_grid_grammar(tokenizer, strings: tuple[str, ...] | None = None) -> GridGrammar:
    from .protocol import REPORT_STRINGS

    strings = strings or REPORT_STRINGS
    sequences = []
    trie: dict = {}
    for index, text in enumerate(strings):
        ids = tokenize_text(tokenizer, text).squeeze(0).tolist()
        sequences.append(ids)
        node = trie
        for token in ids:
            node = node.setdefault(token, {})
        node["__terminal__"] = index
    return GridGrammar(
        strings=tuple(strings),
        values=torch.tensor([float(text) for text in strings], dtype=torch.float32),
        sequences=sequences,
        trie=trie,
        max_depth=max(len(ids) for ids in sequences),
    )


# ---------------------------------------------------------------------------
# Trie walk against a retained cache (differentiable)
# ---------------------------------------------------------------------------
def _select_cache_rows(cache, parent_indices: list[int]):
    """Reorder/duplicate cache rows in place (differentiable indexing)."""

    indices = torch.tensor(parent_indices, dtype=torch.long)
    cache.batch_select_indices(indices)
    return cache


def _suffix_forward(model, cache, prefix_len: int, suffix_ids: torch.Tensor, *, requires_grad: bool):
    """Continue the suffix against ``cache`` (the cache is mutated in place)."""

    total = prefix_len + suffix_ids.shape[1]
    attention = torch.ones((suffix_ids.shape[0], total), dtype=torch.long, device=suffix_ids.device)
    cache_position = torch.arange(prefix_len, total, device=suffix_ids.device)
    if requires_grad:
        output = model(
            input_ids=suffix_ids, attention_mask=attention, past_key_values=cache,
            cache_position=cache_position, use_cache=True,
        )
    else:
        with torch.no_grad():
            output = model(
                input_ids=suffix_ids, attention_mask=attention, past_key_values=cache,
                cache_position=cache_position, use_cache=True,
            )
    return output.logits[:, -1, :], output.past_key_values


def _step_cache(model, cache, prefix_len: int, total_len: int, tokens: list[int], *, requires_grad: bool):
    device = next(model.parameters()).device
    ids = torch.tensor([[token] for token in tokens], dtype=torch.long, device=device)
    attention = torch.ones((ids.shape[0], total_len + 1), dtype=torch.long, device=ids.device)
    tail = total_len + 1
    cache_position = torch.tensor([tail - 1], dtype=torch.long, device=ids.device).expand(ids.shape[0])

    def forward():
        return model(
            input_ids=ids, attention_mask=attention, past_key_values=cache,
            cache_position=cache_position, use_cache=True,
        )

    if requires_grad:
        output = forward()
    else:
        with torch.no_grad():
            output = forward()
    return output.logits[:, -1, :], output.past_key_values


def grid_distribution_on_cache(
    model,
    prefix_cache,
    prefix_len: int,
    suffix_ids: torch.Tensor,
    grammar: GridGrammar,
    *,
    requires_grad: bool,
) -> dict:
    """Distribution over the grammar strings from an answer-fork prefix cache."""

    logits, cache = _suffix_forward(
        model, prefix_cache, prefix_len, suffix_ids, requires_grad=requires_grad
    )
    total_len = prefix_len + suffix_ids.shape[1]
    rows: list[tuple[int, ...]] = [()]
    nodes: list[dict] = [grammar.trie]
    cumulative: list[torch.Tensor] = [torch.zeros((), device=next(model.parameters()).device)]
    completed: dict[int, torch.Tensor] = {}
    while rows:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        new_rows: list[tuple[int, ...]] = []
        new_nodes: list[dict] = []
        new_cumulative: list[torch.Tensor] = []
        parent_indices: list[int] = []
        step_tokens: list[int] = []
        for row_index, node in enumerate(nodes):
            for token in grammar.children(node):
                child = node[token]
                new_rows.append(rows[row_index] + (token,))
                new_nodes.append(child)
                # Keep tensors: casting to float() would detach the graph.
                new_cumulative.append(cumulative[row_index] + log_probs[row_index, token])
                parent_indices.append(row_index)
                step_tokens.append(token)
                terminal = grammar.terminal(child)
                if terminal is not None:
                    completed[terminal] = new_cumulative[-1]
        if not new_rows:
            break
        if all(grammar.terminal(node) is not None for node in new_nodes):
            break
        # Branch the cache per child and process one new token per row.
        branched = _select_cache_rows(cache, parent_indices)
        logits, cache = _step_cache(
            model, branched, prefix_len, total_len, step_tokens, requires_grad=requires_grad
        )
        total_len += 1
        rows, nodes, cumulative = new_rows, new_nodes, new_cumulative
    device = next(model.parameters()).device
    missing = torch.tensor(float("-inf"), device=device)
    logprob_vector = torch.stack(
        [completed[i] if i in completed else missing for i in range(len(grammar.strings))]
    )
    if not torch.isfinite(logprob_vector).any():
        raise ValueError("grid walk produced no complete strings")
    probabilities = torch.softmax(logprob_vector, dim=-1)
    return {
        "probs": probabilities,
        "logprobs": logprob_vector,
        "values": grammar.values,
        "emitted_index": int(torch.argmax(logprob_vector).item()),
    }


# ---------------------------------------------------------------------------
# Prefix handling per arm
# ---------------------------------------------------------------------------
def prefix_with_gating(
    model,
    input_ids: torch.Tensor,
    adapters,
    *,
    adapters_on_prefix: bool,
    requires_grad: bool,
):
    attention = torch.ones_like(input_ids)
    with adapters_enabled(adapters, adapters_on_prefix):
        if requires_grad:
            output = model(
                input_ids=input_ids, attention_mask=attention, use_cache=True
            )
        else:
            with torch.no_grad():
                output = model(
                    input_ids=input_ids, attention_mask=attention, use_cache=True
                )
    return output.past_key_values


def answer_policy(
    model,
    prefix_cache,
    prefix_len: int,
    suffix_ids: torch.Tensor,
    adapters,
    *,
    enabled: bool,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Answer-fork policy from a fresh prefix cache (the cache is mutated)."""

    with adapters_enabled(adapters, enabled):
        logits, _ = _suffix_forward(
            model, prefix_cache, prefix_len, suffix_ids, requires_grad=requires_grad
        )
    return F.softmax(logits.float(), dim=-1)


# ---------------------------------------------------------------------------
# Objectives
# ---------------------------------------------------------------------------
def forecast_loss(probs: torch.Tensor, draws: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Proper score over the emitted distribution: mean_k sum_g p_g (g - Y_k)^2."""

    values = values.to(device=probs.device, dtype=probs.dtype)
    draws = draws.to(device=probs.device, dtype=probs.dtype)
    error = (values[None, :] - draws[:, None]) ** 2
    return (probs[None, :] * error).sum(dim=-1).mean()


def preservation_kl(adapted: torch.Tensor, frozen: torch.Tensor) -> torch.Tensor:
    return (adapted * ((adapted + 1e-9).log() - (frozen + 1e-9).log())).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Training one arm
# ---------------------------------------------------------------------------
@dataclass
class ArmConfig:
    arm: str = "A"
    steps: int = 600
    warmup_steps: int = 0
    share_after_warmup: bool = True
    lr: float = 1e-4
    beta_answer: float = 1.0
    seed: int = 0
    log_every: int = 100

    def __post_init__(self):
        if self.arm not in ("A", "B", "C"):
            raise ValueError(f"unknown arm: {self.arm!r}")
        if self.warmup_steps < 0 or self.warmup_steps > self.steps:
            raise ValueError("warmup_steps must lie in [0, steps]")


def train_arm(
    env,
    *,
    reporter_adapters,
    prefix_and_suffix: list[dict],
    grammar: GridGrammar,
    config: ArmConfig,
) -> dict:
    """Train one arm on precomputed ``(question, variant, draws)`` items.

    ``prefix_and_suffix`` items carry: ``input_ids`` (question prefix),
    ``suffix_ids``, ``draws`` (tensor of independently graded outcomes), and
    the variant adapters to install.  Items are shuffled with ``config.seed``.
    """

    parameters = reporter_parameters(env.model)
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    generator = np.random.default_rng(config.seed)
    order = np.arange(len(prefix_and_suffix))
    losses: list[float] = []
    for step in range(config.steps):
        if step % len(order) == 0:
            generator.shuffle(order)
        item = prefix_and_suffix[int(order[step % len(order)])]
        adapters_on_prefix = config.arm in ("B", "C") and (
            not config.share_after_warmup or step >= config.warmup_steps
        )
        with _variant_installed(env, item):
            prefix_cache = prefix_with_gating(
                env.model,
                item["input_ids"],
                reporter_adapters,
                adapters_on_prefix=adapters_on_prefix,
                requires_grad=adapters_on_prefix,
            )
            with adapters_enabled(reporter_adapters, True):
                distribution = grid_distribution_on_cache(
                    env.model, prefix_cache, item["input_ids"].shape[1], item["suffix_ids"],
                    grammar, requires_grad=True,
                )
            values = grammar.values.to(distribution["probs"].device)
            if step < config.warmup_steps:
                # Probability-teacher warm start: pull the expected report
                # toward the observed success mean before proper-score
                # training against individual draws.
                expected = (distribution["probs"] * values).sum()
                loss = (expected - item["draws"].mean().to(expected.device)) ** 2
            else:
                loss = forecast_loss(distribution["probs"], item["draws"], values)
            if config.arm in ("B", "C") and config.beta_answer > 0:
                # Fresh prefix caches: the report walk mutates its cache.
                adapted_cache = prefix_with_gating(
                    env.model,
                    item["input_ids"],
                    reporter_adapters,
                    adapters_on_prefix=True,
                    requires_grad=True,
                )
                adapted = answer_policy(
                    env.model, adapted_cache, item["input_ids"].shape[1],
                    item["suffix_ids"], reporter_adapters,
                    enabled=True, requires_grad=True,
                )
                frozen_cache = prefix_with_gating(
                    env.model,
                    item["input_ids"],
                    reporter_adapters,
                    adapters_on_prefix=False,
                    requires_grad=False,
                )
                frozen = answer_policy(
                    env.model, frozen_cache, item["input_ids"].shape[1],
                    item["suffix_ids"], reporter_adapters,
                    enabled=False, requires_grad=False,
                )
                loss = loss + config.beta_answer * preservation_kl(adapted, frozen)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if config.log_every and step % config.log_every == 0:
            print(f"  arm {config.arm} step {step}/{config.steps} loss {losses[-1]:.4f}", flush=True)
    return {"arm": config.arm, "steps": config.steps, "final_loss": losses[-1] if losses else None}


class _variant_installed:
    """Temporarily install a variant's frozen adapters around a training item."""

    def __init__(self, env, item):
        self.env = env
        self.item = item
        self.adapters = None

    def __enter__(self):
        from .variants import install_variant

        if self.item.get("state_path") is not None:
            self.adapters = install_variant(
                self.env,
                layers=self.item["layers"],
                rank=self.item["rank"],
                state_path=self.item["state_path"],
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        from .variants import remove_variant

        if self.adapters is not None:
            remove_variant(self.env, self.adapters)
        return None


# ---------------------------------------------------------------------------
# Emitted-number evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def emit_report(env, item, grammar: GridGrammar, *, arm: str, reporter_adapters) -> dict:
    """Greedy constrained decoding of an actual number for one item."""

    with _variant_installed(env, item):
        prefix_cache = prefix_with_gating(
            env.model,
            item["input_ids"],
            reporter_adapters,
            adapters_on_prefix=arm in ("B", "C"),
            requires_grad=False,
        )
        with adapters_enabled(reporter_adapters, True):
            from .reporters import greedy_constrained_report

            from .episodes import PrefixRun

            state = PrefixRun(
                cache=prefix_cache,
                last_hidden=torch.empty(0),
                last_logits=torch.empty(0),
                input_ids=item["input_ids"],
                attention_mask=torch.ones_like(item["input_ids"]),
                prefix_len=int(item["input_ids"].shape[1]),
            )
            report = greedy_constrained_report(
                env.model, state, item["suffix_ids"], env.tokenizer
            )
    return {"string": report.string, "value": report.value, "valid": report.valid}
