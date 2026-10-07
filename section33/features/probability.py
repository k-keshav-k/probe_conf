"""Probability features (P baseline) computed from the model's own logits.

Track A uses the next-token distribution at the last prompt token.  Track B
additionally uses the sequence statistics of the candidate's own tokens.  No
feature uses the reference answer, the reference probability, the reference
rank, object popularity, or the reference alias count.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

TOP_K_MASSES = (10, 50)


def next_token_features(logits: torch.Tensor, position: str) -> dict[str, float]:
    """Top-1, margin, entropy, and fixed top-k mass at one position."""

    probability = torch.softmax(logits.float(), dim=-1)
    top2 = probability.topk(2)
    entropy = -(probability * (probability + 1e-12).log()).sum()
    out = {
        f"prob__top1__{position}": float(top2.values[0]),
        f"prob__top2__{position}": float(top2.values[1]),
        f"prob__margin__{position}": float(top2.values[0] - top2.values[1]),
        f"prob__entropy__{position}": float(entropy),
    }
    for k in TOP_K_MASSES:
        mass = probability.topk(min(k, probability.numel())).values.sum()
        out[f"prob__top{k}_mass__{position}"] = float(mass)
    return out


def sequence_features(
    logits: torch.Tensor,
    token_ids: list[int],
    position: str,
) -> dict[str, float]:
    """Sequence log-probability statistics for the candidate tokens.

    The caller passes one distribution per candidate token: ``logits[t]`` is the
    next-token distribution at the position that produced ``token_ids[t]``
    (the distribution for the first candidate token comes from the last prompt
    position).  All ``n`` tokens are scored, including the first.
    """

    n = len(token_ids)
    if n == 0:
        return {}
    shifted = logits[:n].float()
    targets = torch.tensor(token_ids, device=shifted.device, dtype=torch.long)
    log_probability = F.log_softmax(shifted, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    token_entropy = -(torch.softmax(shifted, dim=-1) * F.log_softmax(shifted, dim=-1)).sum(-1)
    return {
        f"prob__seq_logprob_sum__{position}": float(log_probability.sum()),
        f"prob__seq_logprob_mean__{position}": float(log_probability.mean()),
        f"prob__seq_logprob_min__{position}": float(log_probability.min()),
        f"prob__seq_entropy_mean__{position}": float(token_entropy.mean()),
        f"prob__seq_length__{position}": float(n),
    }
