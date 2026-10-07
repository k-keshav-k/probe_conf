"""CPU checks for hook semantics, causal positions, and ID alignment."""
import unittest

import numpy as np
import pandas as pd
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from scripts.collect import HookCapture
from scripts.probe_eval import align


class SignalContract(unittest.TestCase):
    def test_align_permuted_ids(self):
        feats = pd.DataFrame({"id": [30, 10, 20]})
        labels = pd.DataFrame({"id": [20, 30, 10], "value": [2, 3, 1]})
        raw = {"ids": np.array([10, 20, 30]), "sel": np.array([[1], [2], [3]])}
        df, result = align(feats, labels, raw)
        self.assertEqual(df.value.tolist(), [3, 1, 2])
        self.assertEqual(result["sel"].ravel().tolist(), [3, 1, 2])

    def test_align_rejects_missing_or_duplicate_ids(self):
        feats = pd.DataFrame({"id": [1, 2]})
        labels = pd.DataFrame({"id": [1, 2]})
        for ids in ([1, 1], [1, 3]):
            with self.assertRaises(ValueError):
                align(feats, labels, {"ids": np.array(ids), "sel": np.zeros((2, 1))})

    def test_hooks_and_teacher_forced_position(self):
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                          num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2)
        model = Qwen2ForCausalLM(cfg).eval()
        layers = model.model.layers
        cap = HookCapture(layers, [3])
        observed = {}
        handle = layers[0].mlp.register_forward_pre_hook(
            lambda module, args: observed.update(x=args[0].detach().clone()))
        cap.register()
        with torch.no_grad():
            gold_pass = model(torch.tensor([[4, 5, 6, 7]]), output_hidden_states=True, use_cache=False)
        cap.remove()
        handle.remove()
        x = observed["x"][:, [2, 3]]
        mlp = layers[0].mlp
        g = mlp.gate_proj(x)
        s = mlp.act_fn(g) * mlp.up_proj(x)
        w = mlp.down_proj(s)
        for key, value in [("gate_pre", g), ("sel", s), ("down", w)]:
            self.assertTrue(torch.equal(cap.captured[0][key], value.to(torch.float16)))
        with torch.no_grad():
            prompt_pass = model(torch.tensor([[4, 5, 6]]), output_hidden_states=True, use_cache=False)
            changed_gold = model(torch.tensor([[4, 5, 6, 8]]), output_hidden_states=True, use_cache=False)
        for a, b in zip(gold_pass.hidden_states, prompt_pass.hidden_states):
            torch.testing.assert_close(a[:, 2], b[:, 2], atol=1e-6, rtol=1e-5)
        self.assertFalse(torch.allclose(gold_pass.hidden_states[1][:, 3], changed_gold.hidden_states[1][:, 3]))
        self.assertEqual(len(gold_pass.hidden_states), 3)
        torch.testing.assert_close(gold_pass.hidden_states[-1], model.model.norm(layers[-1](
            gold_pass.hidden_states[-2],
            position_embeddings=model.model.rotary_emb(gold_pass.hidden_states[-2], torch.arange(4).unsqueeze(0)),
            attention_mask=torch.triu(torch.full((1, 1, 4, 4), torch.finfo(torch.float32).min), diagonal=1),
            position_ids=torch.arange(4).unsqueeze(0))), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
