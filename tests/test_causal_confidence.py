"""Contract, leakage, and objective checks for the causal-confidence package.

These tests implement the substantive implementation checks required by the
plan's section 14:

- cache/full-run equivalence,
- no reporter-to-answer mutation,
- frozen parameter/gradient checks,
- exact grid expectation versus exhaustive enumeration,
- policy-gradient agreement with exact gradients on a tiny categorical,
- correct Brier array shapes,
- split isolation,
- absence of gold/condition leakage.
"""

import copy
import os
import types
import unittest

import numpy as np
import torch

# Tiny CPU MLP steps suffer severe thread-pool overhead on this host; cap
# threads before the heavy objective/head fits run.
os.environ.setdefault("OMP_NUM_THREADS", "4")
torch.set_num_threads(min(4, os.cpu_count() or 1))

from transformers import Qwen2Config, Qwen2ForCausalLM

from causal_confidence import interventions as iv
from causal_confidence.baselines import (
    BaselineFeatures,
    baseline_predictions,
    fit_predict_augmented,
    question_features,
)
from causal_confidence.data import (
    Question,
    audit_references,
    build_partitions,
    check_partition_isolation,
    dedupe_questions,
    leakage_scan,
    load_manifest,
    write_manifest,
)
from causal_confidence.episodes import (
    adapters_enabled,
    cache_full_run_equivalence,
    check_prefix_immutability,
    continue_teacher_forced,
    duplicate_run_tolerance,
    frozen_parameter_check,
    generate_answer,
    last_layer_cache_miss,
    run_prefix,
)
from causal_confidence.evaluate import (
    ClusterItem,
    brier,
    cluster_bootstrap,
    contrast_error,
    expected_calibration_error,
    gate_decision,
    mixture_weights,
    sign_agreement,
)
from causal_confidence.objectives import (
    combined_expected_loss,
    enumerate_expected_loss,
    grid_moments,
    leave_one_out_baseline,
    mean_report_objective,
    pair_type_weights,
)
from causal_confidence.p0 import run_p0_controls, synthetic_rows
from causal_confidence.protocol import CONFIDENCE_GRID, REPORT_STRINGS
from causal_confidence.reporters import (
    base_parameter_snapshot,
    enable_reporter_gradients,
    fit_grid_head,
    fit_scalar_head,
    freeze_answer_policy,
    greedy_constrained_report,
    grid_logprobs,
    install_report_adapters,
    remove_report_adapters,
    reporter_parameters,
    snapshot_matches,
)


def tiny_model(vocab_size: int = 64, n_layers: int = 4):
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return Qwen2ForCausalLM(config).eval()


class FakeTokenizer:
    """Character tokenizer for grammar tests with a tiny random model."""

    name_or_path = "char_tokenizer"
    eos_token_id = 63
    pad_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = [ord(character) % 64 for character in text]
        return types.SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    def encode(self, text, add_special_tokens=False):
        return [ord(character) % 64 for character in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(token)) for token in ids)


class ObjectiveContracts(unittest.TestCase):
    def test_exact_expectation_matches_enumeration(self):
        torch.manual_seed(3)
        probs_a = torch.softmax(torch.randn(4, len(CONFIDENCE_GRID)), dim=-1)
        probs_b = torch.softmax(torch.randn(4, len(CONFIDENCE_GRID)), dim=-1)
        target_a = torch.rand(4)
        target_b = torch.rand(4)
        mu_a, var_a = grid_moments(probs_a)
        mu_b, var_b = grid_moments(probs_b)
        for index in range(4):
            formula = combined_expected_loss(
                mu_a[index : index + 1],
                var_a[index : index + 1],
                mu_b[index : index + 1],
                var_b[index : index + 1],
                target_a[index : index + 1],
                target_b[index : index + 1],
                lam=1.0,
            )
            exact = enumerate_expected_loss(
                probs_a[index : index + 1],
                probs_b[index : index + 1],
                target_a[index : index + 1],
                target_b[index : index + 1],
                lam=1.0,
            )
            for key in ("point", "pair", "total"):
                self.assertAlmostEqual(float(formula[key]), exact[key], places=6)

    def test_variance_terms_change_objective(self):
        torch.manual_seed(4)
        probs = torch.softmax(torch.randn(2, len(CONFIDENCE_GRID)), dim=-1) * 0.9
        probs = probs + 0.01
        probs = probs / probs.sum(dim=-1, keepdim=True)
        mu, var = grid_moments(probs)
        target = torch.tensor([0.5, 0.5])
        with_var = combined_expected_loss(mu, var, mu, var, target, target, lam=1.0)
        without = mean_report_objective(mu, mu, target, target, lam=1.0)
        self.assertGreater(
            float(with_var["total"]), float(without["total"]), "variance terms must increase the loss"
        )

    def test_policy_gradient_agreement(self):
        torch.manual_seed(5)
        logits = torch.tensor([0.4, -0.2, 0.1], requires_grad=True)
        rewards = torch.tensor([1.0, 0.0, 0.3])

        def expected_reward(theta):
            return (torch.softmax(theta, dim=-1) * rewards).sum()

        expected_reward(logits).backward()
        exact = logits.grad.detach().clone()
        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=-1)
        estimator = torch.zeros_like(exact)
        for action in range(len(rewards)):
            onehot = torch.zeros_like(exact)
            onehot[action] = 1.0
            estimator += probs[action] * rewards[action] * (onehot - probs)
        self.assertLess(float((estimator - exact).abs().max()), 1e-6)
        # A state-independent baseline must not change the expected gradient.
        baseline = 0.4
        with_baseline = torch.zeros_like(exact)
        for action in range(len(rewards)):
            onehot = torch.zeros_like(exact)
            onehot[action] = 1.0
            with_baseline += probs[action] * (rewards[action] - baseline) * (onehot - probs)
        self.assertLess(float((with_baseline - exact).abs().max()), 1e-6)

    def test_leave_one_out_baseline(self):
        rewards = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
        baseline = leave_one_out_baseline(rewards)
        manual = torch.tensor([[1.0 / 3, 2.0 / 3, 2.0 / 3, 1.0 / 3]])
        self.assertTrue(torch.allclose(baseline, manual))

    def test_pair_type_weights(self):
        pairs = [
            ("target_strong", "clean"),
            ("target_strong", "clean"),
            ("matched_random", "clean"),
            ("target_mild", "target_strong"),
        ]
        weights = pair_type_weights(pairs)
        self.assertAlmostEqual(float(weights.sum()), 0.5, places=6)
        self.assertAlmostEqual(float(weights[0]), float(weights[1]))
        self.assertEqual(float(weights[3]), 0.0)


class EpisodeContracts(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
        self.suffix = torch.tensor([[12, 13]])
        self.positions = [[self.ids.shape[1] - 1]]

    def test_middle_layer_cache_equivalence(self):
        group = iv.ChannelGroup(layer=1, channels=tuple(range(4)))
        result = cache_full_run_equivalence(
            self.model,
            self.ids,
            self.suffix,
            hook_specs=[(group, "attenuate", 0.5)],
            positions=self.positions,
        )
        floor = cache_full_run_equivalence(
            self.model, self.ids, self.suffix, hook_specs=[], positions=[]
        )
        tolerance = max(10 * floor["max_abs_diff"], 1e-4)
        self.assertLessEqual(result["max_abs_diff"], tolerance)
        # The intervention must actually change the full run.
        clean = cache_full_run_equivalence(
            self.model, self.ids, self.suffix, hook_specs=[], positions=[]
        )
        with torch.no_grad():
            intervened = run_prefix(
                self.model,
                self.ids,
                hooks=iv.InterventionHooks(
                    self.model, [(group, "attenuate", 0.5)], self.positions
                ),
                capture_hidden=False,
            )
        self.assertGreater(
            float((intervened.last_logits - run_prefix(self.model, self.ids).last_logits).abs().max()),
            1e-5,
        )
        self.assertTrue(clean["max_abs_diff"] <= tolerance)

    def test_last_layer_hook_misses_cache(self):
        result = last_layer_cache_miss(
            self.model,
            self.ids,
            self.suffix,
            layer_index=self.model.config.num_hidden_layers - 1,
            positions=self.positions,
            channels=tuple(range(4)),
        )
        self.assertFalse(result["cache_reaches"])

    def test_prefix_isolation(self):
        prefix = run_prefix(self.model, self.ids)
        result = check_prefix_immutability(self.model, prefix, self.suffix)
        self.assertTrue(result["cache_unchanged"])
        self.assertEqual(result["branch_repeat_max_abs_diff"], 0.0)

    def test_duplicate_tolerance(self):
        result = duplicate_run_tolerance(self.model, self.ids, runs=2)
        self.assertEqual(result["max_logit_difference"], 0.0)

    def test_second_draw_does_not_mutate_first(self):
        prefix = run_prefix(self.model, self.ids)
        tokenizer = FakeTokenizer()
        first = generate_answer(
            self.model, prefix, self.suffix, tokenizer, max_new_tokens=2, seed=1
        )
        second = generate_answer(
            self.model, prefix, self.suffix, tokenizer, max_new_tokens=2, seed=2
        )
        third = generate_answer(
            self.model, prefix, self.suffix, tokenizer, max_new_tokens=2, seed=1
        )
        self.assertEqual(first.token_ids, third.token_ids)
        self.assertEqual(prefix.cache.get_seq_length(), self.ids.shape[1])
        self.assertIsNotNone(second.token_ids)


class AdapterContracts(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.ids = torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]])
        self.suffix = torch.tensor([[12, 13]])
        self.prefix = run_prefix(self.model, self.ids)

    def test_disabled_adapters_change_nothing(self):
        adapters = install_report_adapters(self.model, (1,), rank=4)
        freeze_answer_policy(self.model)
        with adapters_enabled(adapters, False):
            disabled = continue_teacher_forced(self.model, self.prefix, self.suffix)
        baseline = continue_teacher_forced(self.model, self.prefix, self.suffix)
        self.assertTrue(torch.equal(disabled, baseline))

    def test_trained_adapter_changes_suffix_only_when_enabled(self):
        adapters = install_report_adapters(self.model, (1,), rank=4)
        freeze_answer_policy(self.model)
        with torch.no_grad():
            for adapter in adapters:
                adapter.lora_b.add_(0.1)
        with adapters_enabled(adapters, True):
            enabled = continue_teacher_forced(self.model, self.prefix, self.suffix)
        with adapters_enabled(adapters, False):
            disabled = continue_teacher_forced(self.model, self.prefix, self.suffix)
        self.assertGreater(float((enabled - disabled).abs().max()), 1e-6)

    def test_reporter_step_freezes_answer_policy(self):
        adapters = install_report_adapters(self.model, (1,), rank=4)
        freeze_answer_policy(self.model)
        enable_reporter_gradients(adapters)
        snapshot = base_parameter_snapshot(self.model)
        optimizer = torch.optim.Adam(reporter_parameters(self.model), lr=1e-3)
        target = torch.tensor([[int(ord("0") % 64), int(ord(".") % 64), int(ord("5") % 64)]])

        def step(model):
            with adapters_enabled(adapters, True):
                logits = continue_teacher_forced(
                    model, self.prefix, target, requires_grad=True
                )
            loss = torch.nn.functional.cross_entropy(logits[0, :-1], target[0, 1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        result = frozen_parameter_check(self.model, ("lora_",), step)
        self.assertTrue(result["passed"])
        self.assertTrue(all("lora" in name for name in result["changed_trainable"]))
        self.assertTrue(snapshot_matches(self.model, snapshot))

    def test_remove_adapters_restores_base_modules(self):
        original_class = type(self.model.model.layers[1].mlp.gate_proj)
        adapters = install_report_adapters(self.model, (1,), rank=4)
        self.assertIsInstance(self.model.model.layers[1].mlp.gate_proj, type(adapters[0]))
        remove_report_adapters(self.model, adapters)
        self.assertIsInstance(self.model.model.layers[1].mlp.gate_proj, original_class)


class InterventionContracts(unittest.TestCase):
    def test_attenuation_scales_only_selected_channels(self):
        s = torch.ones(1, 2, 8)
        rows = torch.tensor([0])
        cols = torch.tensor([1])
        out = iv.apply_attenuation(s, (2, 4), 0.5, rows=rows, cols=cols)
        self.assertAlmostEqual(float(out[0, 1, 2]), 0.5)
        self.assertAlmostEqual(float(out[0, 1, 4]), 0.5)
        self.assertAlmostEqual(float(out[0, 0, 2]), 1.0)

    def test_restore_replaces_exact_values(self):
        s = torch.randn(1, 2, 8)
        rows = torch.tensor([0])
        cols = torch.tensor([1])
        values = torch.tensor([[[9.0, -3.0]]])
        out = iv.restore(s, (0, 3), values, rows=rows, cols=cols)
        self.assertAlmostEqual(float(out[0, 1, 0]), 9.0)
        self.assertAlmostEqual(float(out[0, 1, 3]), -3.0)
        self.assertAlmostEqual(float(out[0, 0, 0]), float(s[0, 0, 0]))

    def test_reinsert_undoes_attenuation(self):
        s = torch.randn(2, 3, 16)
        rows = torch.tensor([0, 1, 0, 1, 0, 1])
        cols = torch.tensor([0, 0, 1, 1, 2, 2])
        channels = (3, 5)
        attenuated = iv.apply_attenuation(s, channels, 0.4, rows=rows, cols=cols)
        removed = s[rows, cols][:, list(channels)] * 0.4
        restored = iv.reinsert(attenuated, channels, removed, rows=rows, cols=cols)
        self.assertTrue(torch.allclose(restored, s))

    def test_removed_norm_matches_direct_computation(self):
        s = torch.randn(1, 2, 8)
        weight = torch.randn(5, 8)
        rows = torch.tensor([0])
        cols = torch.tensor([1])
        norm = iv.removed_contribution_norm(s, weight, (1, 3), 0.5, rows=rows, cols=cols)
        removed = 0.5 * s[rows, cols][:, [1, 3]] @ weight[:, [1, 3]].T
        self.assertAlmostEqual(float(norm[0]), float(removed.norm()))

    def test_position_normalization(self):
        rows, cols = iv.normalize_positions([[5], [7, 8]], 2, torch.device("cpu"))
        self.assertEqual(rows.tolist(), [0, 1, 1])
        self.assertEqual(cols.tolist(), [5, 7, 8])
        rows, cols = iv.normalize_positions(4, 3, torch.device("cpu"))
        self.assertEqual(rows.tolist(), [0, 1, 2])
        self.assertEqual(cols.tolist(), [4, 4, 4])

    def test_norm_matched_group_size_and_exclusion(self):
        profile = torch.linspace(0.0, 1.0, 64)
        target = (10, 20, 30)
        group = iv.norm_matched_channels(profile, target, 3, exclude=(40, 41))
        self.assertEqual(len(group), 3)
        self.assertFalse(set(group) & set(target))
        self.assertFalse(set(group) & {40, 41})

    def test_candidate_sites_exclude_last_layer(self):
        sites = iv.candidate_sites(28)
        self.assertTrue(all(site <= 26 for site in sites))
        self.assertEqual(len(set(sites)), len(sites))


class DataContracts(unittest.TestCase):
    def make_rows(self):
        rows = []
        for subject in range(40):
            for relation in ("father", "place_of_birth"):
                rows.append(
                    Question(
                        question_id=f"q{subject}_{relation}",
                        subject_id=f"s{subject}",
                        subject_name=f"Subject {subject}",
                        relation=relation,
                        question_text=f"Question for {subject} about {relation}?",
                        references=(f"answer_{subject}",),
                        source="test",
                    )
                )
        return rows

    def test_dedupe_and_audit(self):
        rows = self.make_rows()
        duplicate = Question(
            question_id="duplicate",
            subject_id="s0",
            subject_name="Subject 0",
            relation="father",
            question_text="Another phrasing?",
            references=("extra_alias",),
            source="other",
        )
        kept, report = dedupe_questions(rows + [duplicate])
        self.assertEqual(report["duplicate_rows"], 1)
        merged = next(row for row in kept if row.subject_id == "s0" and row.relation == "father")
        self.assertIn("extra_alias", merged.references)
        audit = audit_references(kept)
        self.assertTrue(audit["passed"])

    def test_partitions_are_subject_disjoint(self):
        rows = self.make_rows()
        sizes = {"discovery": 4, "validation": 6, "train": 8, "dev": 4, "test": 8, "transfer": 4}
        partitions, report = build_partitions(rows, sizes, seed=0)
        overlap = check_partition_isolation(partitions)
        self.assertEqual(overlap, {})
        self.assertFalse(report["reduced"])
        self.assertLessEqual(len({row.question_id for row in rows}), sum(report["achieved"].values()))

    def test_reduced_pool_requires_flag(self):
        rows = self.make_rows()[:2]
        sizes = {"discovery": 100, "validation": 100, "train": 100, "dev": 100, "test": 100, "transfer": 100}
        with self.assertRaises(ValueError):
            build_partitions(rows, sizes, seed=0)
        partitions, report = build_partitions(rows, sizes, seed=0, allow_reduced=True)
        self.assertTrue(report["reduced"])

    def test_manifest_round_trip(self):
        rows = self.make_rows()
        sizes = {"discovery": 4, "validation": 6, "train": 8, "dev": 4, "test": 8, "transfer": 4}
        partitions, _ = build_partitions(rows, sizes, seed=0)
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            payload = write_manifest(partitions, path)
            loaded = load_manifest(path)
            self.assertEqual(
                sum(len(items) for items in loaded.values()),
                sum(len(items) for items in partitions.values()),
            )
            self.assertIn("manifest_hash", payload)

    def test_leakage_scan(self):
        clean = leakage_scan(["Who is the father of X?", "Answer:"], ["Charles", "target_strong"])
        self.assertTrue(clean["clean"])
        dirty = leakage_scan(["the answer is Charles"], ["Charles"])
        self.assertFalse(dirty["clean"])
        self.assertEqual(dirty["hits"][0]["needle"], "Charles")


class EvaluateContracts(unittest.TestCase):
    def test_brier_shapes(self):
        scores = np.array([0.2, 0.8, 0.5])
        outcomes = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 1]])
        value = brier(scores, outcomes)
        manual = np.mean(
            [((scores[index] - row) ** 2).mean() for index, row in enumerate(outcomes)]
        )
        self.assertAlmostEqual(value, float(manual))
        self.assertAlmostEqual(brier(np.full(4, 0.5), np.zeros((4, 3))), 0.25)

    def test_brier_weights(self):
        scores = np.array([0.0, 1.0])
        outcomes = np.array([[0.0], [0.0]])
        weighted = brier(scores, outcomes, weights=np.array([1.0, 3.0]))
        self.assertAlmostEqual(weighted, 0.75)

    def test_contrast_error(self):
        value = contrast_error([0.1, 0.9], [0.6, 0.2], [0.1, 0.9], [0.6, 0.2])
        self.assertAlmostEqual(value, 0.0, places=9)
        nonzero = contrast_error([0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.6, 0.2])
        self.assertAlmostEqual(nonzero, 0.02, places=9)

    def test_mixture_weights(self):
        weights = mixture_weights(["clean", "matched_random"])
        self.assertAlmostEqual(weights.sum(), 0.6)

    def test_cluster_bootstrap_deterministic(self):
        items = [
            ClusterItem(subject_id=f"s{index // 2}", payload={"value": index})
            for index in range(8)
        ]
        metric = lambda index, all_items: float(  # noqa: E731
            np.mean([all_items[i].payload["value"] for i in index])
        )
        first = cluster_bootstrap(items, metric, n_samples=200, seed=0)
        second = cluster_bootstrap(items, metric, n_samples=200, seed=0)
        self.assertEqual(first["estimate"], second["estimate"])
        self.assertEqual(first["ci_low"], second["ci_low"])

    def test_calibration_and_sign_helpers(self):
        scores = np.linspace(0.05, 0.95, 40)
        outcomes = (np.random.default_rng(0).random((40, 20)) < scores[:, None]).astype(float)
        self.assertGreaterEqual(expected_calibration_error(scores, outcomes), 0.0)
        agreement = sign_agreement([1, -1, 1, -1], [1, -1, -1, 1], min_effect=0.0)
        self.assertAlmostEqual(agreement["agreement"], 0.5)

    def test_gate_decision(self):
        passed = gate_decision(0.06, 0.01, 0.05)
        self.assertTrue(passed["passed"])
        failed = gate_decision(0.06, -0.01, 0.05)
        self.assertFalse(failed["passed"])


class ReporterContracts(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model(vocab_size=64, n_layers=2)
        self.tokenizer = FakeTokenizer()
        self.ids = torch.tensor([[4, 5, 6, 7, 8]])
        self.prefix = run_prefix(self.model, self.ids)
        self.suffix = torch.tensor([[9, 10]])

    def test_greedy_constrained_report_is_in_grammar(self):
        adapters = install_report_adapters(self.model, (0,), rank=4)
        freeze_answer_policy(self.model)
        with adapters_enabled(adapters, True):
            report = greedy_constrained_report(
                self.model, self.prefix, self.suffix, self.tokenizer
            )
        self.assertIn(report.string, REPORT_STRINGS)
        self.assertTrue(report.valid)

    def test_grid_logprobs_normalize(self):
        adapters = install_report_adapters(self.model, (0,), rank=4)
        freeze_answer_policy(self.model)
        with adapters_enabled(adapters, True):
            result = grid_logprobs(self.model, self.prefix, self.suffix, self.tokenizer)
        self.assertEqual(len(result["probs"]), len(CONFIDENCE_GRID))
        self.assertAlmostEqual(sum(result["probs"]), 1.0, places=5)

    def test_scalar_and_grid_heads(self):
        torch.manual_seed(0)
        x = torch.randn(200, 5)
        y = torch.rand(200)
        scalar = fit_scalar_head(x, y, steps=100)
        grid = fit_grid_head(x, y, steps=100)
        with torch.no_grad():
            scalar_values = scalar(x)
            grid_values = grid.expected(x)
        self.assertEqual(scalar_values.shape, (200,))
        self.assertEqual(grid_values.shape, (200,))
        self.assertTrue(torch.all(scalar_values >= 0) and torch.all(scalar_values <= 1))
        self.assertLess(float((grid_values - y).pow(2).mean()), 0.1)


class BaselineContracts(unittest.TestCase):
    def test_baseline_predictions_and_added_value(self):
        rng = np.random.default_rng(0)
        n = 120
        question = question_features(
            ["father"] * n, [100.0] * n, ["Who is the father of X?"] * n
        )
        prefix = rng.normal(size=(n, 4)).astype(np.float32)
        outcomes = (rng.random((n, 8)) < 0.6).astype(np.float32)
        report = np.clip(0.5 + 0.3 * prefix[:, 0], 0.0, 1.0).astype(np.float32)
        train = BaselineFeatures(blocks={"question": question[:80], "prefix": prefix[:80]})
        test = BaselineFeatures(blocks={"question": question[80:], "prefix": prefix[80:]})
        predictions = baseline_predictions(train, outcomes[:80], test, seed=0)
        self.assertIn("constant_half", predictions)
        self.assertIn("prefix", predictions)
        self.assertEqual(len(predictions["prefix"]), 40)
        augmented = fit_predict_augmented(
            train,
            outcomes[:80],
            test,
            report[:80],
            report[80:],
            baseline="prefix",
            seed=0,
        )
        self.assertEqual(augmented.shape, (40,))


class P0Contracts(unittest.TestCase):
    def test_synthetic_positive_control_passes(self):
        rows = synthetic_rows(n_train=3_000, n_test=1_500, seed=0)
        result = run_p0_controls(rows, seed=0, head_steps=800)
        self.assertIn("gates", result)
        self.assertTrue(result["gates"]["reproduction_mse"]["passed"])
        self.assertTrue(result["gates"]["same_input_brier"]["passed"])


if __name__ == "__main__":
    unittest.main()
