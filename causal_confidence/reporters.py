"""Scalar diagnostics and the native numerical reporting adapter.

Two families live here:

- External heads over saved features: a direct scalar sigmoid regression and a
  grid-mean head over the 21 report values.  These are feasibility diagnostics
  (P0 and P3 scalar checks); they are not native verbal reporters.
- The native reporter: rank-8 adapters on selected MLP gate/up projections,
  active only on the report suffix.  The emitted number is produced by greedy
  constrained decoding over the fixed grammar ``0.00, 0.05, ..., 1.00``;
  the training objective uses the exact distribution over grammar strings.

A deployment-shaped report is the number actually emitted.  An expected number
computed from token probabilities is a diagnostic only (plan section 5).
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .episodes import (
    PrefixRun,
    _suffix_forward,
    _step_token,
    adapters_enabled,
    tokenize_text,
)
from .objectives import grid_moments, grid_tensor
from .protocol import CONFIDENCE_GRID, PROTOCOL, REPORT_STRINGS


# ---------------------------------------------------------------------------
# External heads (feasibility diagnostics)
# ---------------------------------------------------------------------------
class ScalarReportHead(nn.Module):
    """Two-layer MLP with a sigmoid scalar output over saved features."""

    def __init__(self, in_features: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        modules: list[nn.Module] = [nn.Linear(in_features, hidden), nn.GELU()]
        for _ in range(depth - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.GELU()])
        modules.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


class GridReportHead(nn.Module):
    """Two-layer MLP with a softmax distribution over the report grid."""

    def __init__(self, in_features: int, hidden: int = 256, depth: int = 2):
        super().__init__()
        modules: list[nn.Module] = [nn.Linear(in_features, hidden), nn.GELU()]
        for _ in range(depth - 1):
            modules.extend([nn.Linear(hidden, hidden), nn.GELU()])
        modules.append(nn.Linear(hidden, len(REPORT_STRINGS)))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

    @torch.no_grad()
    def expected(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.forward(x)
        grid = grid_tensor(probs.device, probs.dtype)
        return probs @ grid


def fit_scalar_head(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int = 0,
    steps: int = 1_500,
    lr: float = 1e-3,
    batch: int = 256,
    hidden: int = 256,
    weight_decay: float = 1e-4,
) -> ScalarReportHead:
    """Train a scalar head by squared error (outcome means or continuous targets).

    A small weight decay is the P0-tested regularization: outcome draws are
    noisy and an unregularized head overfits them relative to a regularized
    probability baseline.
    """

    torch.manual_seed(seed)
    head = ScalarReportHead(x.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        index = torch.randint(0, len(x), (min(batch, len(x)),), generator=generator)
        prediction = head(x[index])
        loss = (prediction - y[index]).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    head.eval()
    return head


def fit_grid_head(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int = 0,
    steps: int = 1_500,
    lr: float = 1e-3,
    batch: int = 256,
    hidden: int = 256,
    variance_weight: float = 0.0,
    weight_decay: float = 1e-4,
) -> GridReportHead:
    """Train a grid head on outcome means.

    With ``variance_weight=0`` this minimizes squared error of the expected
    report (the risk form used by the toy); a positive weight adds the
    variance term of the exact expected-loss objective.  A small weight decay
    is the P0-tested regularization against noisy outcome means.
    """

    torch.manual_seed(seed)
    head = GridReportHead(x.shape[1], hidden=hidden)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    grid = grid_tensor(x.device, x.dtype)
    for _ in range(steps):
        index = torch.randint(0, len(x), (min(batch, len(x)),), generator=generator)
        probs = head(x[index])
        mean, variance = grid_moments(probs, grid)
        loss = (mean - y[index]).pow(2).mean() + variance_weight * variance.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    head.eval()
    return head


# ---------------------------------------------------------------------------
# Native adapter
# ---------------------------------------------------------------------------
class LinearAdapter(nn.Module):
    """Frozen linear projection plus a trainable low-rank gate/up adapter."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = 1.0 / rank
        # Adapters are inactive unless a report-branch context enables them.
        self.enabled = False
        self.parent_ref: tuple[nn.Module, str] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        if self.enabled:
            # Adapter parameters stay in float32; compute in float32 and cast
            # the update back so fp16 models keep their dtype.
            update = (x.float() @ self.lora_a.T) @ self.lora_b.T
            output = output + self.scale * update.to(output.dtype)
        return output

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features


def install_report_adapters(
    model, layer_indices: tuple[int, ...], rank: int = PROTOCOL.report_rank
) -> list[LinearAdapter]:
    """Wrap gate/up projections with rank-8 adapters on the selected layers.

    The base projections are frozen inside the adapter.  Every other model
    parameter is left untouched; the caller freezes them with
    :func:`freeze_answer_policy` before reporter training.
    """

    layers = _decoder_layers(model)
    adapters: list[LinearAdapter] = []
    for index in layer_indices:
        mlp = layers[index].mlp
        for name in ("gate_proj", "up_proj"):
            base = getattr(mlp, name)
            adapter = LinearAdapter(base, rank)
            adapter = adapter.to(device=base.weight.device)
            adapter.parent_ref = (mlp, name)
            setattr(mlp, name, adapter)
            adapters.append(adapter)
    return adapters


def remove_report_adapters(model, adapters: list[LinearAdapter]) -> None:
    """Restore the frozen base projections; the caller keeps the adapters."""

    for adapter in adapters:
        if adapter.parent_ref is None:
            continue
        parent, name = adapter.parent_ref
        setattr(parent, name, adapter.base)
        adapter.parent_ref = None


def _decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("cannot locate transformer layers")


def freeze_answer_policy(model) -> None:
    """Freeze every parameter; reporter adapters are re-enabled separately."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)


def enable_reporter_gradients(adapters: list[LinearAdapter]) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for adapter in adapters:
        adapter.lora_a.requires_grad_(True)
        adapter.lora_b.requires_grad_(True)
        parameters.extend([adapter.lora_a, adapter.lora_b])
    return parameters


def reporter_parameters(model) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in model.named_parameters()
        if "lora_a" in name or "lora_b" in name
    ]


def base_parameter_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if "lora_a" not in name and "lora_b" not in name
    }


def snapshot_matches(model, snapshot: dict[str, torch.Tensor]) -> bool:
    for name, value in snapshot.items():
        if not torch.equal(value, model.get_parameter(name).detach()):
            return False
    return True


# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------
def string_token_sequences(tokenizer, strings: tuple[str, ...] = REPORT_STRINGS) -> list[list[int]]:
    sequences = []
    for value in strings:
        ids = tokenize_text(tokenizer, value).squeeze(0).tolist()
        if not ids:
            raise ValueError(f"string {value!r} tokenizes to nothing")
        sequences.append(ids)
    return sequences


def build_trie(sequences: list[list[int]]) -> dict:
    root: dict = {}
    for sequence in sequences:
        node = root
        for token in sequence:
            node = node.setdefault(token, {})
        node["__terminal__"] = True
    return root


def trie_terminals(node: dict) -> bool:
    return bool(node.get("__terminal__"))


def grid_value(string: str) -> float:
    value = round(float(string), 2)
    if not any(abs(value - grid) < 1e-9 for grid in CONFIDENCE_GRID):
        raise ValueError(f"report string {string!r} outside the grammar")
    return value


@dataclass
class GreedyReport:
    string: str
    value: float
    token_ids: list[int]
    valid: bool
    stop_reason: str


def greedy_constrained_report(
    model,
    prefix: PrefixRun,
    suffix_ids: torch.Tensor,
    tokenizer,
    *,
    strings: tuple[str, ...] = REPORT_STRINGS,
) -> GreedyReport:
    """Greedy decoding restricted to the report grammar."""

    sequences = string_token_sequences(tokenizer, strings)
    trie = build_trie(sequences)
    cache = copy.deepcopy(prefix.cache)
    output = _suffix_forward(model, cache, prefix, suffix_ids)
    cache = output.past_key_values
    total_len = prefix.prefix_len + suffix_ids.shape[1]
    node = trie
    generated: list[int] = []
    stop_reason = "grammar"
    while not trie_terminals(node):
        allowed = [token for token in node if token != "__terminal__"]
        if not allowed:
            stop_reason = "dead_end"
            break
        logits = output.logits[0, -1]
        allowed_logits = logits[allowed]
        token = allowed[int(torch.argmax(allowed_logits).item())]
        generated.append(token)
        output = _step_token(model, cache, token, total_len, suffix_ids.device)
        cache = output.past_key_values
        total_len += 1
        node = node[token]
    string = tokenizer.decode(generated, skip_special_tokens=True).strip()
    valid = string in set(strings)
    return GreedyReport(
        string=string if valid else string,
        value=float(string) if valid else float("nan"),
        token_ids=generated,
        valid=valid,
        stop_reason=stop_reason,
    )


@torch.no_grad()
def grid_logprobs(
    model,
    prefix: PrefixRun,
    suffix_ids: torch.Tensor,
    tokenizer,
    *,
    strings: tuple[str, ...] = REPORT_STRINGS,
) -> dict:
    """Sequence log-probabilities over every grammar string.

    Each string is teacher-forced after the report suffix from a private cache
    copy.  This is the exact distribution the sample-based objective needs.
    """

    return grid_logprobs_train(
        model, prefix, suffix_ids, tokenizer, strings=strings, requires_grad=False
    )


def grid_logprobs_train(
    model,
    prefix: PrefixRun,
    suffix_ids: torch.Tensor,
    tokenizer,
    *,
    strings: tuple[str, ...] = REPORT_STRINGS,
    requires_grad: bool = True,
) -> dict:
    """Differentiable version used by the native reporter objective.

    Returns differentiable ``logprob_tensor``/``probs_tensor`` plus detached
    lists for logging.  The grid distribution is the softmax over full-string
    sequence log-probabilities (numbers may have different token lengths).
    """

    sequences = string_token_sequences(tokenizer, strings)
    totals = []
    for sequence in sequences:
        ids = torch.tensor([sequence], device=suffix_ids.device)
        full_suffix = torch.cat([suffix_ids, ids[:, :-1]], dim=1)
        logits = _suffix_forward_logits(
            model, prefix, full_suffix, requires_grad=requires_grad
        )
        window = logits[0, suffix_ids.shape[1] - 1 : suffix_ids.shape[1] - 1 + len(sequence)]
        log_probs = F.log_softmax(window.float(), dim=-1)
        total = torch.zeros((), device=suffix_ids.device)
        for position, token in enumerate(sequence):
            total = total + log_probs[position, token]
        totals.append(total)
    stacked = torch.stack(totals)
    probabilities = torch.softmax(stacked, dim=-1)
    return {
        "strings": list(strings),
        "logprob_tensor": stacked,
        "probs_tensor": probabilities,
        "logprobs": stacked.detach().tolist(),
        "probs": probabilities.detach().tolist(),
    }


def _suffix_forward_logits(
    model, prefix: PrefixRun, suffix_ids: torch.Tensor, *, requires_grad: bool = False
):
    cache = copy.deepcopy(prefix.cache)
    output = _suffix_forward(model, cache, prefix, suffix_ids, requires_grad=requires_grad)
    return output.logits


def report_distribution_tensor(grid_logprob_result: dict, device=None) -> torch.Tensor:
    return torch.tensor(grid_logprob_result["probs"], device=device)


def expected_report_value(grid_logprob_result: dict) -> float:
    probs = torch.tensor(grid_logprob_result["probs"])
    grid = grid_tensor()
    return float((probs * grid).sum())


def sample_report_strings(
    grid_logprob_result: dict,
    *,
    samples: int,
    seed: int,
    temperature: float = 1.0,
) -> list[dict]:
    """Sample emitted report strings from the independent report distribution.

    Each state's report depends on its own distribution only (plan section 8.1).
    """

    generator = torch.Generator().manual_seed(seed)
    weights = torch.tensor(grid_logprob_result["probs"]).clamp_min(1e-12).pow(1.0 / temperature)
    weights = weights / weights.sum()
    index = torch.multinomial(weights, samples, replacement=True, generator=generator)
    strings = grid_logprob_result["strings"]
    return [
        {"index": int(i), "string": strings[int(i)], "value": float(strings[int(i)])}
        for i in index
    ]


# ---------------------------------------------------------------------------
# Format warm start
# ---------------------------------------------------------------------------
def format_warm_start(
    model,
    adapters: list[LinearAdapter],
    examples: list[tuple[PrefixRun, torch.Tensor, str]],
    tokenizer,
    *,
    steps: int = 200,
    lr: float = 1e-4,
    seed: int = 0,
) -> dict:
    """Shared format warm start on identical examples for every arm.

    ``examples`` is a list of ``(prefix, suffix_ids, target_string)``.  The
    loss is next-token cross-entropy over the target string tokens; no outcome
    labels or condition identifiers are used.
    """

    parameters = enable_reporter_gradients(adapters)
    optimizer = torch.optim.Adam(parameters, lr=lr)
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    with adapters_enabled(adapters, True):
        for _ in range(steps):
            prefix, suffix_ids, target = examples[int(torch.randint(len(examples), (1,), generator=generator))]
            ids = torch.tensor(
                [string_token_sequences(tokenizer, (target,))[0]], device=suffix_ids.device
            )
            full = torch.cat([suffix_ids, ids], dim=1)
            logits = _suffix_forward_logits(model, prefix, full, requires_grad=True)
            start = suffix_ids.shape[1] - 1
            window = logits[0, start : start + ids.shape[1] - 1]
            loss = F.cross_entropy(window, ids.squeeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
    return {"steps": steps, "final_loss": losses[-1] if losses else None}
