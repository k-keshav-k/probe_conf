"""MLP hook contracts and verification for the installed models.

Plan section 3.1 and section 8.2 require:

- verification that the three MLP projections have no bias,
- reconstruction control: ``W_down s`` must equal the module output,
- capture of the normalized MLP input ``x`` and, when asked, the module output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..features import mlp_math as mm


def get_layers(model):
    """Return the transformer layer list for a Qwen2 or Mistral model."""

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("cannot locate transformer layers on this model")


def mlp_modules(layer):
    """Return ``(gate_proj, up_proj, down_proj)`` for Qwen2 or Mistral."""

    mlp = layer.mlp
    return mlp.gate_proj, mlp.up_proj, mlp.down_proj


def verify_mlp_contract(model, atol: float = 5e-3) -> dict:
    """Check biases and reconstruction for every layer on a random batch.

    The check runs one forward pass with a small random input.  At every layer
    it compares the captured module output with
    ``W_down * (SiLU(W_gate x) * W_up x)``.
    """

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    hidden = model.config.hidden_size
    layers = get_layers(model)
    checks: dict[str, object] = {"n_layers": len(layers), "biases": {}, "reconstruction": {}}

    for index, layer in enumerate(layers):
        gate, up, down = mlp_modules(layer)
        checks["biases"][index] = {
            "gate": gate.bias is not None,
            "up": up.bias is not None,
            "down": down.bias is not None,
        }

    captured: dict[int, dict[str, torch.Tensor]] = {index: {} for index in range(len(layers))}
    handles = []

    for index, layer in enumerate(layers):
        gate, _, down = mlp_modules(layer)

        def pre_hook(module, args, index=index):
            captured[index]["x"] = args[0].detach()

        def out_hook(module, args, output, index=index):
            captured[index]["o"] = output.detach()

        handles.append(gate.register_forward_pre_hook(pre_hook))
        handles.append(down.register_forward_hook(out_hook))

    try:
        with torch.no_grad():
            dummy = torch.randn(1, 4, hidden, device=device, dtype=dtype)
            model(
                inputs_embeds=dummy,
                attention_mask=torch.ones(1, 4, device=device, dtype=torch.long),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    per_layer_error = {}
    for index, layer in enumerate(layers):
        gate, up, down = mlp_modules(layer)
        x = captured[index]["x"].float()
        _, _, _, reconstructed = mm.mlp_activations(
            x, gate.weight.float(), up.weight.float(), down.weight.float()
        )
        per_layer_error[index] = float(
            (reconstructed - captured[index]["o"].float()).detach().abs().max()
        )
    max_error = max(per_layer_error.values())
    checks["reconstruction"] = {
        "max_abs_error": max_error,
        "per_layer_max_error": per_layer_error,
        "passed": max_error <= atol,
    }
    checks["bias_free"] = all(
        not any(flags.values()) for flags in checks["biases"].values()
    )
    checks["passed"] = bool(checks["bias_free"] and checks["reconstruction"]["passed"])
    return checks


@dataclass
class MLPCapture:
    """Capture MLP activations at chosen token positions.

    Keys captured per layer:

    - ``x``: normalized MLP input (input to ``gate_proj``)
    - ``a``: gate projection output
    - ``b``: up projection output
    - ``s``: selection coefficient (input to ``down_proj``)
    - ``o``: module output (only when ``capture_output`` is true)

    ``positions`` is a list with one entry per batch row.  Each entry is a list
    of token indices to capture.  Captured tensors are stored on CPU in the
    model dtype unless ``keep_device`` is set.
    """

    def __init__(
        self,
        model: object,
        positions: list[list[int]],
        capture_output: bool = False,
        keep_device: bool = False,
        keys: tuple[str, ...] = ("x", "a", "b", "s"),
    ):
        self.model = model
        self.positions = positions
        self.capture_output = capture_output
        self.keep_device = keep_device
        self.keys = keys
        self.captured: dict[int, dict[str, torch.Tensor]] = {}
        self.handles: list = []

    def _index(self, tensor: torch.Tensor) -> torch.Tensor:
        batch = tensor.shape[0]
        indices = torch.as_tensor(self.positions, device=tensor.device, dtype=torch.long)
        if indices.shape[0] != batch:
            raise ValueError("positions must match the batch size")
        rows = torch.arange(batch, device=tensor.device).repeat_interleave(indices.shape[1])
        flat = indices.reshape(-1)
        out = tensor[rows, flat].reshape(batch, indices.shape[1], -1).detach()
        if self.keep_device:
            return out
        return out.to("cpu")

    def register(self) -> None:
        for index, layer in enumerate(get_layers(self.model)):
            gate, up, down = mlp_modules(layer)
            self.captured[index] = {}

            def gate_pre_hook(module, args, index=index):
                if "x" in self.keys:
                    self.captured[index]["x"] = self._index(args[0])

            def gate_hook(module, args, output, index=index):
                if "a" in self.keys:
                    self.captured[index]["a"] = self._index(output)

            def up_hook(module, args, output, index=index):
                if "b" in self.keys:
                    self.captured[index]["b"] = self._index(output)

            def down_pre_hook(module, args, index=index):
                if "s" in self.keys:
                    self.captured[index]["s"] = self._index(args[0])

            self.handles.append(gate.register_forward_pre_hook(gate_pre_hook))
            self.handles.append(gate.register_forward_hook(gate_hook))
            self.handles.append(up.register_forward_hook(up_hook))
            self.handles.append(down.register_forward_pre_hook(down_pre_hook))
            if self.capture_output:

                def down_hook(module, args, output, index=index):
                    self.captured[index]["o"] = self._index(output)

                self.handles.append(down.register_forward_hook(down_hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def stack(self, key: str) -> torch.Tensor:
        """Return ``(n_layers, batch, n_positions, d)`` for one captured key."""

        return torch.stack([self.captured[i][key] for i in sorted(self.captured)], dim=0)
