"""Signed channel manipulation, restoration, and matched controls.

The intervention acts on the gated-MLP selection coefficient

    g = W_gate x,  v = W_up x,  s = SiLU(g) * v,  o = W_down s.

For a channel group ``G`` and attenuation ``a`` the hook replaces
``s_j <- (1 - a) s_j`` for ``j in G`` at declared prefix positions.  The signed
value is preserved (the sign of ``s_j`` is untouched); only its magnitude
changes.  The removed contribution is

    sum_j a * s_j * W_down[:, j].

Realized perturbation norms are logged because nominally equal coefficients do
not imply equal interventions (plan section 7.3).

Restoration patches a prespecified downstream group with its clean activations:
``s_j <- s_j^clean`` for ``j in G_rescue`` at the same positions.  The original
disruption stays active.  Disabling the hook or reinserting the whole removed
contribution at the same site are engineering checks, not mechanistic
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .protocol import InterventionConfig


@dataclass(frozen=True)
class ChannelGroup:
    """A fixed channel group at one decoder layer's MLP."""

    layer: int
    channels: tuple[int, ...]
    label: str = "target"
    relation: str | None = None

    def __post_init__(self) -> None:
        if not self.channels:
            raise ValueError("channel group is empty")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channel group contains duplicates")

    @property
    def size(self) -> int:
        return len(self.channels)

    def to_json(self) -> dict:
        return {
            "layer": self.layer,
            "channels": list(self.channels),
            "label": self.label,
            "relation": self.relation,
        }

    @classmethod
    def from_json(cls, payload: dict) -> "ChannelGroup":
        return cls(
            layer=int(payload["layer"]),
            channels=tuple(int(channel) for channel in payload["channels"]),
            label=str(payload.get("label", "target")),
            relation=payload.get("relation"),
        )


# ---------------------------------------------------------------------------
# Group nomination
# ---------------------------------------------------------------------------
def channel_mass_profile(mass: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
    """Mean ``|s_j| · ||W_down[:, j]||`` per channel over selected rows.

    ``mass`` has shape ``(n_rows, n_positions, d_ff)``.  A zero vector is
    returned for an empty selection.
    """

    values = mass if rows is None else mass[rows]
    if values.numel() == 0:
        return torch.zeros(mass.shape[-1], dtype=mass.dtype, device=mass.device)
    if values.ndim == 2:
        return values.mean(dim=0)
    return values.mean(dim=(0, 1))


def top_contribution_channels(
    profile: torch.Tensor, k: int, *, exclude: tuple[int, ...] = ()
) -> tuple[int, ...]:
    """Channels with the largest mean contribution mass."""

    scores = profile.clone().float()
    if exclude:
        scores[list(exclude)] = float("-inf")
    available = int(torch.isfinite(scores).sum().item())
    k = min(k, available)
    if k <= 0:
        raise ValueError("no available channels")
    index = torch.topk(scores, k).indices
    return tuple(sorted(int(i) for i in index))


def relation_specific_channels(
    profiles: dict[str, torch.Tensor],
    relation: str,
    k: int,
    *,
    exclude: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Channels whose mass is largest for one relation relative to the others."""

    if relation not in profiles:
        raise KeyError(f"no mass profile for relation {relation!r}")
    others = [profile for name, profile in profiles.items() if name != relation]
    if not others:
        return top_contribution_channels(profiles[relation], k, exclude=exclude)
    contrast = profiles[relation].float() - torch.stack(
        [profile.float() for profile in others]
    ).mean(dim=0)
    return top_contribution_channels(contrast.clamp_min(0), k, exclude=exclude)


def norm_matched_channels(
    profile: torch.Tensor,
    target: tuple[int, ...],
    k: int,
    *,
    exclude: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Count-matched random group with the closest sorted mass profile.

    Candidate channels are ranked by their mean mass; the target group's sorted
    values are matched greedily to their nearest available ranks.  This is a
    norm-matching heuristic, not exact moment matching; achieved norms are
    reported with the result.
    """

    excluded = set(target) | set(exclude)
    candidates = [i for i in range(profile.shape[0]) if i not in excluded]
    if len(candidates) < k:
        raise ValueError("not enough candidate channels for a matched random group")
    pool = sorted(candidates, key=lambda i: float(profile[i]))
    target_sorted = sorted(float(profile[i]) for i in target)
    chosen: list[int] = []
    for value in target_sorted:
        nearest = min(pool, key=lambda i: abs(float(profile[i]) - value))
        chosen.append(nearest)
        pool.remove(nearest)
    return tuple(sorted(chosen))


def matched_random_rank(
    profile: torch.Tensor,
    target_sorted: list[float],
    k: int,
    *,
    seed: int,
    exclude: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """A count- and approximately norm-matched random group.

    The pool of eligible channels is restricted to those whose mean mass lies
    in the min/max range of the target group, then ``k`` are drawn at random
    with ``seed``.  Several draws can be compared on the observed perturbation
    norm before one is frozen.
    """

    excluded = set(exclude)
    low, high = min(target_sorted), max(target_sorted)
    eligible = [
        i
        for i in range(profile.shape[0])
        if i not in excluded and low <= float(profile[i]) <= high
    ]
    if len(eligible) < k:
        eligible = [i for i in range(profile.shape[0]) if i not in excluded]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(eligible), generator=generator).tolist()
    return tuple(sorted(eligible[i] for i in order[:k]))


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------
def normalize_positions(positions, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return flat ``(row_index, token_index)`` tensors for scattered updates.

    Accepted forms: an int (same token for every row), a flat list of ints
    (same tokens for every row), or a list of per-row token lists.
    """

    if isinstance(positions, int):
        per_row: list[list[int]] = [[positions] for _ in range(batch_size)]
    elif torch.is_tensor(positions):
        values = positions.tolist()
        if values and isinstance(values[0], list):
            per_row = [[int(v) for v in row] for row in values]
        else:
            per_row = [[int(v) for v in values] for _ in range(batch_size)]
    elif isinstance(positions, (list, tuple)):
        values = list(positions)
        if values and isinstance(values[0], (list, tuple)):
            per_row = [[int(v) for v in row] for row in values]
        else:
            per_row = [[int(v) for v in values] for _ in range(batch_size)]
    else:
        raise TypeError(f"unsupported positions type: {type(positions)!r}")
    if len(per_row) == 1 and batch_size > 1:
        per_row = [list(per_row[0]) for _ in range(batch_size)]
    if len(per_row) != batch_size:
        raise ValueError(f"positions rows {len(per_row)} != batch {batch_size}")
    rows: list[int] = []
    cols: list[int] = []
    for row, tokens in enumerate(per_row):
        for token in tokens:
            rows.append(row)
            cols.append(token)
    return (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(cols, dtype=torch.long, device=device),
    )


# ---------------------------------------------------------------------------
# Pure transforms
# ---------------------------------------------------------------------------
def _channel_index(s: torch.Tensor, channels: tuple[int, ...]) -> torch.Tensor:
    return torch.as_tensor(channels, dtype=torch.long, device=s.device)


def apply_attenuation(
    s: torch.Tensor,
    channels: tuple[int, ...],
    attenuation: float,
    *,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    """Scale ``s`` by ``1 - attenuation`` on the selected channels/positions."""

    if not 0.0 <= attenuation <= 1.0:
        raise ValueError("attenuation must lie in [0, 1]")
    out = s.clone()
    if rows.numel() == 0:
        return out
    column = _channel_index(s, channels)
    block = out[rows, cols].clone()
    index = column.expand(block.shape[0], -1)
    updated = block.gather(1, index) * (1.0 - attenuation)
    block = block.scatter(1, index, updated)
    out[rows, cols] = block
    return out


def removed_contribution_norm(
    s: torch.Tensor,
    weight: torch.Tensor,
    channels: tuple[int, ...],
    attenuation: float,
    *,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    """Norm of the removed vector ``sum_j a·s_j·W_down[:,j]`` per (row, pos)."""

    selection = s[rows, cols][:, list(channels)]
    basis = weight[:, list(channels)]
    removed = (selection * attenuation) @ basis.T
    return removed.norm(dim=-1)


def _selection_values(values: torch.Tensor, n_selected: int) -> torch.Tensor:
    """Flatten an activation payload to ``(n_selected, n_channels)``."""

    if values.dim() == 3:
        values = values.reshape(-1, values.shape[-1])
    elif values.dim() == 1:
        values = values.unsqueeze(0)
    if values.shape[0] != n_selected:
        raise ValueError(
            f"activation payload rows {values.shape[0]} != selected positions {n_selected}"
        )
    return values


def reinsert(
    s: torch.Tensor,
    channels: tuple[int, ...],
    values: torch.Tensor,
    *,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    """Add ``values`` back to the selected channels (direct-undo check)."""

    out = s.clone()
    if rows.numel() == 0:
        return out
    column = _channel_index(s, channels)
    block = out[rows, cols].clone()
    index = column.expand(block.shape[0], -1)
    block = block.scatter_add(
        1, index, _selection_values(values.to(block.dtype), block.shape[0])
    )
    out[rows, cols] = block
    return out


def restore(
    s: torch.Tensor,
    channels: tuple[int, ...],
    clean_values: torch.Tensor,
    *,
    rows: torch.Tensor,
    cols: torch.Tensor,
) -> torch.Tensor:
    """Replace the selected channels with clean activations."""

    out = s.clone()
    if rows.numel() == 0:
        return out
    column = _channel_index(s, channels)
    block = out[rows, cols].clone()
    index = column.expand(block.shape[0], -1)
    block = block.scatter(
        1, index, _selection_values(clean_values.to(block.dtype), block.shape[0])
    )
    out[rows, cols] = block
    return out


# ---------------------------------------------------------------------------
# Hook installation
# ---------------------------------------------------------------------------
def _decoder_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("cannot locate transformer layers on this model")


def _mlp_down(layer):
    return layer.mlp.down_proj


def make_transform_hook(transform, positions):
    """Wrap a pure transform ``(s, rows, cols) -> s'`` as a down_proj pre-hook.

    Rows and columns are derived from the actual activation shape, so the same
    hook works for any batch size.
    """

    def hook(module, args):
        s = args[0]
        if s is None:
            return None
        rows, cols = normalize_positions(positions, s.shape[0], s.device)
        if rows.numel() == 0:
            return None
        return (transform(s, rows, cols),)

    return hook


class InterventionHooks:
    """Context manager installing one or more MLP channel hooks.

    ``specs`` is a list of ``(group, kind, payload)``:

    - ``kind="attenuate"``: ``payload`` is the attenuation.
    - ``kind="restore"``: ``payload`` is a ``(n_positions, n_channels)`` tensor
      of clean activations for each call (broadcast over the batch), or a
      ``(batch, n_positions, n_channels)`` tensor.
    - ``kind="reinsert"``: like ``restore`` but the values are added.

    Realized perturbation norms for each attenuation are accumulated in
    ``removed_norms`` keyed by ``group.label``.
    """

    def __init__(self, model, specs, positions, *, record_norms: bool = True):
        self.model = model
        self.specs = list(specs)
        self.positions = positions
        self.record_norms = record_norms
        self.removed_norms: dict[str, list[torch.Tensor]] = {}
        self.handles: list = []

    @property
    def active(self) -> bool:
        return bool(self.handles)

    def __enter__(self) -> "InterventionHooks":
        if self.active:
            return self
        layers = _decoder_layers(self.model)
        for group, kind, payload in self.specs:
            down = _mlp_down(layers[group.layer])
            weight = down.weight.detach()

            def transform(
                s,
                rows,
                cols,
                _group=group,
                _kind=kind,
                _payload=payload,
                _weight=weight,
                _self=self,
            ):
                if _kind == "attenuate":
                    out = apply_attenuation(
                        s, _group.channels, float(_payload), rows=rows, cols=cols
                    )
                    if _self.record_norms:
                        norm = removed_contribution_norm(
                            s, _weight, _group.channels, float(_payload), rows=rows, cols=cols
                        )
                        _self.removed_norms.setdefault(_group.label, []).append(
                            norm.detach()
                        )
                    return out
                values = _payload
                if torch.is_tensor(values) and values.dim() == 2:
                    values = values.unsqueeze(0).expand(s.shape[0], -1, -1)
                if _kind == "restore":
                    return restore(s, _group.channels, values, rows=rows, cols=cols)
                if _kind == "reinsert":
                    return reinsert(s, _group.channels, values, rows=rows, cols=cols)
                raise ValueError(f"unknown intervention kind: {_kind!r}")

            self.handles.append(
                down.register_forward_pre_hook(make_transform_hook(transform, self.positions))
            )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for name, values in list(self.removed_norms.items()):
            if isinstance(values, list) and values and torch.is_tensor(values[0]):
                self.removed_norms[name] = torch.cat(list(values), dim=0)
        return None

    def perturbation_summary(self) -> dict:
        summary = {}
        for name, values in self.removed_norms.items():
            tensor = values if torch.is_tensor(values) else torch.cat(list(values), dim=0)
            summary[name] = {
                "n_observations": int(tensor.numel()),
                "mean": float(tensor.mean()) if tensor.numel() else 0.0,
                "max": float(tensor.max()) if tensor.numel() else 0.0,
            }
        return summary


# ---------------------------------------------------------------------------
# Clean activation capture
# ---------------------------------------------------------------------------
@torch.no_grad()
def capture_selection(
    model,
    input_ids: torch.Tensor,
    layer_index: int,
    positions,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the full selection coefficient ``s`` at one layer.

    Shape ``(n_selected, d_ff)`` where ``n_selected`` is the total number of
    (row, position) pairs.  Used to build channel-mass profiles.
    """

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    captured: dict[str, torch.Tensor] = {}
    layers = _decoder_layers(model)
    down = _mlp_down(layers[layer_index])

    def hook(module, args):
        s = args[0]
        rows, cols = normalize_positions(positions, s.shape[0], s.device)
        captured["values"] = s[rows, cols].detach().clone()

    handle = down.register_forward_pre_hook(hook)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return captured["values"]


def selection_mass(s: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``|s_j| · ||W_down[:, j]||`` per position (plan section 7.1)."""

    norms = weight.norm(dim=0)
    return s.abs() * norms


@torch.no_grad()
def capture_group_activations(
    model,
    input_ids: torch.Tensor,
    group: ChannelGroup,
    positions,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return clean ``s`` values at the group's channels.

    Shape ``(n_selected, group.size)`` with rows ordered by the per-row
    position lists.  The caller must pass the same positions used at
    intervention time.
    """

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    captured: dict[str, torch.Tensor] = {}
    layers = _decoder_layers(model)
    down = _mlp_down(layers[group.layer])

    def hook(module, args):
        s = args[0]
        rows, cols = normalize_positions(positions, s.shape[0], s.device)
        captured["values"] = s[rows, cols][:, list(group.channels)].detach().clone()

    handle = down.register_forward_pre_hook(hook)
    try:
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        handle.remove()
    return captured["values"]


def candidate_sites(n_layers: int, config: InterventionConfig = InterventionConfig()) -> tuple[int, ...]:
    """Three middle-depth locations at ~1/3, 1/2, and 2/3 of decoder depth.

    The last decoder layer is excluded by construction: a hook there cannot
    reach a continuation cache (plan section 5).
    """

    last_allowed = n_layers - 2
    sites: list[int] = []
    for fraction in config.site_fractions:
        site = int(round(fraction * (n_layers - 1)))
        site = max(0, min(site, last_allowed))
        if site not in sites:
            sites.append(site)
    return tuple(sites)
