"""Small gated-MLP decoder with gate/up adapters and a report head.

The base model is trained during the shared warm start.  After that, the
comparison freezes every base parameter and the down projections, and trains
only the gate/up adapters (proposal sections 5.2 and 6.5).  The report head is
trainable only during reporting updates.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ToyConfig


class LinearAdapter(nn.Module):
    """Frozen linear map plus a trainable low-rank gate/up adapter."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = 1.0 / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * (x @ self.lora_a.T) @ self.lora_b.T


class Block(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(config.d_model, config.n_heads, batch_first=True)
        self.ln_mlp = nn.LayerNorm(config.d_model)
        self.gate = LinearAdapter(
            nn.Linear(config.d_model, config.d_hidden, bias=False), config.adapter_rank
        )
        self.up = LinearAdapter(
            nn.Linear(config.d_model, config.d_hidden, bias=False), config.adapter_rank
        )
        self.down = nn.Linear(config.d_hidden, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.ln_attn(x)
        attended, _ = self.attn(normed, normed, normed, need_weights=False)
        hidden = x + attended
        mlp_input = self.ln_mlp(hidden)
        channels = F.silu(self.gate(mlp_input)) * self.up(mlp_input)
        return hidden + self.down(channels)


class Selector(nn.Module):
    def __init__(self, config: ToyConfig):
        super().__init__()
        self.config = config
        self.memory_proj = nn.Linear(config.memory_dim, config.d_model, bias=False)
        self.relation_emb = nn.Embedding(config.n_relations, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_final = nn.LayerNorm(config.d_model)
        self.answer_head = nn.Linear(config.d_model, config.n_answers, bias=False)
        modules: list[nn.Module] = [
            nn.Linear(config.d_model, config.d_report),
            nn.GELU(),
        ]
        for _ in range(config.report_layers - 1):
            modules.extend([nn.Linear(config.d_report, config.d_report), nn.GELU()])
        modules.append(nn.Linear(config.d_report, len(config.confidence_grid)))
        self.report_head = nn.Sequential(*modules)

    def hidden(self, memory: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack([self.memory_proj(memory), self.relation_emb(relation_ids)], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.ln_final(tokens[:, -1])

    def answer_logits(self, memory: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        return self.answer_head(self.hidden(memory, relation_ids))

    def report_logits(self, memory: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        return self.report_head(self.hidden(memory, relation_ids))

    def policy(self, memory: torch.Tensor, relation_ids: torch.Tensor, temperature: float | None = None):
        temperature = self.config.temperature if temperature is None else temperature
        return torch.softmax(self.answer_logits(memory, relation_ids) / temperature, dim=-1)

    def report_distribution(self, memory: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        return self.report_distribution_from_hidden(self.hidden(memory, relation_ids))

    def report_distribution_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.report_head(hidden), dim=-1)

    @torch.no_grad()
    def expected_report(self, memory: torch.Tensor, relation_ids: torch.Tensor) -> torch.Tensor:
        grid = torch.tensor(
            self.config.confidence_grid, dtype=memory.dtype, device=memory.device
        )
        return self.report_distribution(memory, relation_ids) @ grid


def set_requires_grad(model: nn.Module, substrings: tuple[str, ...] | list[str]) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(substring in name for substring in substrings)


def set_all_trainable(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def set_retrieval_trainable(model: nn.Module) -> None:
    """Freeze the base and down projections; train gate/up adapters only."""

    layers = model.config.adapter_layers or (model.config.n_layers - 1,)
    patterns: list[str] = []
    for layer in layers:
        if not 0 <= layer < model.config.n_layers:
            raise ValueError(f"adapter layer {layer} outside [0, {model.config.n_layers})")
        patterns.extend([f"blocks.{layer}.gate.lora_", f"blocks.{layer}.up.lora_"])
    set_requires_grad(model, patterns)


def set_report_trainable(model: nn.Module) -> None:
    """Freeze the retrieval model; train only the reporting adapter."""

    set_requires_grad(model, ["report_head"])


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
