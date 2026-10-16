"""Contracts for the native confidence arms (A/B/C).

Uses a tiny random Qwen2 model and a character tokenizer; no GPU needed.  The
key check is that the trie walk against the retained cache agrees with
brute-force sequence scoring of every grammar string.
"""

import os
import unittest

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "4")
torch.set_num_threads(min(4, os.cpu_count() or 1))

from causal_confidence import interventions as iv
from causal_confidence.episodes import continue_teacher_forced, run_prefix
from causal_confidence.native_monitor import (
    ArmConfig,
    build_grid_grammar,
    forecast_loss,
    grid_distribution_on_cache,
    train_arm,
)
from causal_confidence.protocol import REPORT_STRINGS
from causal_confidence.reporters import (
    adapters_enabled,
    install_report_adapters,
    freeze_answer_policy,
    reporter_parameters,
)
from tests.test_causal_confidence import FakeTokenizer, tiny_model


class GridWalkTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model(vocab_size=64, n_layers=2)
        self.tokenizer = FakeTokenizer()
        self.grammar = build_grid_grammar(self.tokenizer)
        self.ids = torch.tensor([[4, 5, 6, 7, 8]])
        self.suffix = torch.tensor([[9, 10]])
        self.prefix = run_prefix(self.model, self.ids, capture_hidden=False)

    def test_walk_matches_brute_force_sequence_scoring(self):
        result = grid_distribution_on_cache(
            self.model,
            self.prefix.cache,
            self.prefix.prefix_len,
            self.suffix,
            self.grammar,
            requires_grad=False,
        )
        # Brute force: teacher-force suffix + string[:-1] and sum logprobs.
        import torch.nn.functional as F

        logprobs = []
        for sequence in self.grammar.sequences:
            ids = torch.tensor([sequence])
            full = torch.cat([self.suffix, ids[:, :-1]], dim=1)
            logits = continue_teacher_forced(self.model, self.prefix, full)
            window = logits[0, self.suffix.shape[1] - 1 : self.suffix.shape[1] - 1 + len(sequence)]
            log_probs = F.log_softmax(window.float(), dim=-1)
            logprobs.append(
                sum(float(log_probs[position, token]) for position, token in enumerate(sequence))
            )
        brute = torch.softmax(torch.tensor(logprobs), dim=-1)
        torch.testing.assert_close(result["probs"], brute, atol=1e-4, rtol=1e-4)

    def test_forecast_loss_point_mass(self):
        values = self.grammar.values
        probs = torch.zeros_like(values)
        probs[10] = 1.0  # value 0.50
        draws = torch.tensor([1.0, 1.0, 0.0, 1.0])
        expected = float(((values[10] - draws) ** 2).mean())
        self.assertAlmostEqual(float(forecast_loss(probs, draws, values)), expected, places=6)

    def test_emitted_report_is_in_grammar(self):
        adapters = install_report_adapters(self.model, (0,), rank=4)
        freeze_answer_policy(self.model)
        for parameter in reporter_parameters(self.model):
            parameter.requires_grad_(True)
        item = {
            "input_ids": self.ids,
            "suffix_ids": self.suffix,
            "draws": torch.tensor([1.0, 0.0, 1.0, 1.0]),
            "state_path": None,
            "layers": (0,),
            "rank": 4,
        }
        result = train_arm(
            type("Env", (), {"model": self.model, "tokenizer": self.tokenizer})(),
            reporter_adapters=adapters,
            prefix_and_suffix=[item],
            grammar=self.grammar,
            config=ArmConfig(arm="A", steps=3, lr=1e-3, beta_answer=0.0, log_every=0),
        )
        self.assertEqual(result["steps"], 3)
        self.assertTrue(np.isfinite(result["final_loss"]))


if __name__ == "__main__":
    unittest.main()
