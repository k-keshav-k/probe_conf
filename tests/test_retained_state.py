"""CPU checks for the retained-state fork contract.

These tests use a tiny random Qwen2 model.  They verify the three properties
the retrieval-monitoring plan depends on:

1. A branch from the retained state equals an independent full run.
2. A deep-copied branch cannot change the retained state.
3. An intervention inside the prefix reaches every later branch.
"""

import copy
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from section33.analysis.retained_state import (
    PrefixState,
    additive_transform,
    continue_branch,
    fork_branches,
    run_prefix,
)


def tiny_model():
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return Qwen2ForCausalLM(config).eval()


class RetainedStateContract(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
        self.prefix = self.ids[:, :5]
        self.branches = {
            "answer": self.ids[:, 5:],
            "confidence": torch.tensor([[12, 13]]),
        }

    def _full_run(self, suffix):
        with torch.no_grad():
            return self.model(
                input_ids=torch.cat([self.prefix, suffix], dim=1), use_cache=False
            ).logits[:, -suffix.shape[1] :]

    def test_branch_matches_full_run(self):
        state = run_prefix(self.model, self.prefix)
        outputs = fork_branches(self.model, state, self.branches)
        for name, suffix in self.branches.items():
            torch.testing.assert_close(
                outputs[name].logits, self._full_run(suffix), atol=1e-5, rtol=1e-5
            )

    def test_retained_state_survives_branches(self):
        state = run_prefix(self.model, self.prefix)
        before = state.cache.get_seq_length()
        fork_branches(self.model, state, self.branches)
        self.assertEqual(state.cache.get_seq_length(), before)
        self.assertEqual(state.prefix_len, 5)
        # A second round of branches gives the same result.
        first = fork_branches(self.model, state, self.branches)
        second = fork_branches(self.model, state, self.branches)
        for name in self.branches:
            torch.testing.assert_close(first[name].logits, second[name].logits)

    def test_deepcopy_is_independent(self):
        state = run_prefix(self.model, self.prefix)
        clone = copy.deepcopy(state.cache)
        self.assertEqual(clone.get_seq_length(), state.cache.get_seq_length())
        self.assertIsNot(clone, state.cache)

    def test_intervention_reaches_every_branch(self):
        clean = run_prefix(self.model, self.prefix)
        clean_out = fork_branches(self.model, clean, self.branches)
        vector = torch.zeros(self.model.config.hidden_size)
        vector[0] = 3.0
        intervened = run_prefix(
            self.model,
            self.prefix,
            layer_index=0,
            transform=additive_transform(vector, token_index=-1),
        )
        intervened_out = fork_branches(self.model, intervened, self.branches)
        for name in self.branches:
            difference = (intervened_out[name].logits - clean_out[name].logits).abs().max()
            self.assertGreater(float(difference), 1e-4, msg=name)

    def test_intervention_is_token_specific(self):
        vector = torch.zeros(self.model.config.hidden_size)
        vector[1] = 2.0
        last = run_prefix(
            self.model,
            self.prefix,
            layer_index=0,
            transform=additive_transform(vector, token_index=-1),
        )
        first = run_prefix(
            self.model,
            self.prefix,
            layer_index=0,
            transform=additive_transform(vector, token_index=0),
        )
        last_out = continue_branch(self.model, last, self.branches["answer"])
        first_out = continue_branch(self.model, first, self.branches["answer"])
        self.assertFalse(torch.allclose(last_out.logits, first_out.logits))

    def test_last_layer_hook_does_not_reach_branches(self):
        """A hook on the last decoder layer changes only the fork-point state.

        The continuation reads cached keys and values from earlier layers.  It
        does not read the fork-point hidden state.  A last-layer hook therefore
        cannot reach the branches.  Use a middle layer for interventions.
        """

        clean = run_prefix(self.model, self.prefix)
        clean_out = continue_branch(self.model, clean, self.branches["answer"])
        vector = torch.zeros(self.model.config.hidden_size)
        vector[0] = 3.0
        intervened = run_prefix(
            self.model,
            self.prefix,
            layer_index=self.model.config.num_hidden_layers - 1,
            transform=additive_transform(vector, token_index=2),
        )
        intervened_out = continue_branch(self.model, intervened, self.branches["answer"])
        torch.testing.assert_close(intervened_out.logits, clean_out.logits, atol=1e-6, rtol=1e-5)

    def test_fork_false_mutates_retained_cache(self):
        """The shared-cache path is unsafe for multiple branches.

        The model appends the suffix to the cache in place.  This test encodes
        the reason every branch must get a deep copy.
        """

        state = run_prefix(self.model, self.prefix)
        before = state.cache.get_seq_length()
        continue_branch(self.model, state, self.branches["answer"], fork=False)
        self.assertGreater(state.cache.get_seq_length(), before)


if __name__ == "__main__":
    unittest.main()
