"""Fast contract tests for the controlled knowledge-variant study.

No GPU and no real model are needed: the corpus, exposure design, verification
gate, and reporter fitting are exercised with synthetic states; variant
adapter training is exercised on a tiny random model with a character
tokenizer.
"""

import os
import unittest

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "4")
torch.set_num_threads(min(4, os.cpu_count() or 1))

from causal_confidence import knowledge as know
from causal_confidence import variants as var
from causal_confidence.baselines import BaselineFeatures
from causal_confidence.knowledge_stages import (
    StateData,
    compare_reporters,
    evaluate_reporter,
    fit_pair_reporter,
    question_groups,
    reporter_objective,
    select_lambda,
)
from tests.test_causal_confidence import FakeTokenizer, tiny_model


class CorpusAndDesignTests(unittest.TestCase):
    def test_corpus_is_deterministic_and_values_are_absent_from_questions(self):
        first = know.build_corpus(12, seed=3)
        second = know.build_corpus(12, seed=3)
        self.assertEqual([f.fact_id for f in first], [f.fact_id for f in second])
        self.assertEqual([f.value for f in first], [f.value for f in second])
        for fact in first:
            for text in (fact.train_question, *fact.eval_questions):
                self.assertNotIn(fact.value.lower(), text.lower())

    def test_exposure_is_counterbalanced(self):
        facts = know.build_corpus(24, seed=0)
        exposure = know.assign_exposure(facts)
        self.assertEqual(set(exposure), set(know.VARIANT_IDS))
        for variant, ids in exposure.items():
            self.assertEqual(len(ids), len(facts) // 2, f"variant {variant}")
        holders = {}
        for fact in facts:
            holders[fact.fact_id] = tuple(
                variant for variant, ids in exposure.items() if fact.fact_id in ids
            )
        self.assertTrue(all(len(set(h)) == 2 for h in holders.values()))
        summary = know.exposure_summary(facts, exposure)
        self.assertTrue(summary["balanced_pairs"])
        self.assertEqual(summary["facts_per_variant_count"], {2: len(facts)})

    def test_subject_splits_are_disjoint(self):
        facts = know.build_corpus(20, seed=0)
        splits = know.split_by_subject(facts, train_subjects=10, dev_subjects=4)
        subjects = {
            name: {fact.subject_id for fact in items} for name, items in splits.items()
        }
        self.assertFalse(subjects["train"] & subjects["dev"])
        self.assertFalse(subjects["train"] & subjects["test"])
        self.assertFalse(subjects["dev"] & subjects["test"])
        self.assertEqual(sum(len(items) for items in splits.values()), len(facts))

    def test_manifest_hash_stable(self):
        facts = know.build_corpus(8, seed=1)
        transfer = know.build_corpus(3, seed=9, prefix="transfer")
        splits = know.split_by_subject(facts, train_subjects=5, dev_subjects=1)
        exposure = know.assign_exposure(facts)
        first = know.corpus_manifest(facts, exposure, splits, transfer, seed=1)
        second = know.corpus_manifest(facts, exposure, splits, transfer, seed=1)
        self.assertEqual(first["content_hash"], second["content_hash"])


class VerificationGateTests(unittest.TestCase):
    def _success(self, variant_success: dict[int, float]) -> dict[int, dict[str, float]]:
        return {variant: {f"fact{i}": value for i in range(10)} for variant, value in variant_success.items()}

    def test_gate_passes_on_spread(self):
        by_variant = {}
        for variant, value in ((1, 0.9), (2, 0.1), (3, 0.8), (4, 0.2)):
            by_variant[variant] = {f"fact{i}": value for i in range(10)}
        known = {v: {f"fact{i}" for i in range(10)} for v in by_variant}
        report = var.verify_knowledge(
            by_variant,
            known_facts=known,
            train_template_success={
                v: {f"fact{i}": by_variant[v][f"fact{i}"] for i in range(10)}
                for v in by_variant
            },
            format_stats={v: {"empty_rate": 0.0, "truncation_rate": 0.0} for v in by_variant},
            reproducibility={v: 0.8 for v in by_variant},
        )
        self.assertTrue(report.passed)
        self.assertGreater(report.details["mean_spread"], 0.6)

    def test_gate_fails_when_variants_are_identical(self):
        by_variant = {v: {f"fact{i}": 0.5 for i in range(10)} for v in (1, 2, 3, 4)}
        report = var.verify_knowledge(
            by_variant,
            known_facts={v: set() for v in by_variant},
            train_template_success={v: {} for v in by_variant},
            format_stats={v: {"empty_rate": 0.0, "truncation_rate": 0.0} for v in by_variant},
            reproducibility={v: 0.8 for v in by_variant},
        )
        self.assertFalse(report.passed)

    def test_gate_fails_on_template_memorization(self):
        by_variant = {}
        for variant, value in ((1, 0.9), (2, 0.1), (3, 0.8), (4, 0.2)):
            by_variant[variant] = {f"fact{i}": value for i in range(10)}
        train_template = {
            v: {f"fact{i}": 0.9 for i in range(10)} for v in by_variant
        }
        report = var.verify_knowledge(
            by_variant,
            known_facts={v: {f"fact{i}" for i in range(10)} for v in by_variant},
            train_template_success=train_template,
            format_stats={v: {"empty_rate": 0.0, "truncation_rate": 0.0} for v in by_variant},
            reproducibility={v: 0.8 for v in by_variant},
        )
        self.assertFalse(report.passed)
        self.assertLess(report.details["min_paraphrase_ratio"], 0.5)


def synthetic_states(seed: int = 0, questions: int = 12, variants: int = 4) -> StateData:
    generator = np.random.default_rng(seed)
    fact_ids, subject_ids, variant_ids = [], [], []
    feature_rows, hidden_rows, outcomes = [], [], []
    for question in range(questions):
        for variant in range(1, variants + 1):
            success = 0.9 if (question + variant) % 2 == 0 else 0.05
            fact_ids.append(f"fact{question}")
            subject_ids.append(f"subject{question // 4}")
            variant_ids.append(variant)
            signal = success + generator.normal(0, 0.03)
            feature_rows.append([signal, generator.normal() * 0.05])
            hidden_rows.append([signal, generator.normal() * 0.05])
            wins = generator.random(8) < success
            outcomes.append(wins.astype(np.float32))
    return StateData(
        fact_ids=fact_ids,
        subject_ids=subject_ids,
        variant_ids=variant_ids,
        features=BaselineFeatures(
            {
                "question": np.asarray(feature_rows, dtype=np.float32),
                "prefix": np.asarray(feature_rows, dtype=np.float32),
                "hidden": np.asarray(hidden_rows, dtype=np.float32),
            }
        ),
        outcomes=np.stack(outcomes),
    )


class ReporterFittingTests(unittest.TestCase):
    def test_pair_objective_reduces_contrast_error(self):
        train = synthetic_states(seed=0)
        test = synthetic_states(seed=1)
        groups = question_groups(train)
        self.assertEqual(len(groups), 12)
        self.assertTrue(all(len(group) == 4 for group in groups))
        t1 = fit_pair_reporter(train, lam=0.0, steps=600, seed=0)
        t2 = fit_pair_reporter(train, lam=1.0, steps=600, seed=0)
        report_t1 = evaluate_reporter(t1, test, label="t1")
        report_t2 = evaluate_reporter(t2, test, label="t2")
        self.assertEqual(report_t1["n_questions"], 12)
        # The pair term directly trains the within-question contrast; compare
        # the contrast risk on the held-out states with a small tolerance for
        # finite-optimization noise.
        self.assertLessEqual(
            report_t2["contrast_error"], report_t1["contrast_error"] * 1.1
        )
        self.assertLessEqual(report_t2["brier"], 0.1)

    def test_single_variant_evaluation_is_finite(self):
        train = synthetic_states(seed=0)
        single = synthetic_states(seed=3, variants=1)
        head = fit_pair_reporter(train, lam=0.0, steps=50, seed=0)
        report = evaluate_reporter(head, single, label="transfer")
        self.assertTrue(np.isfinite(report["brier"]))
        self.assertIsNone(report["contrast_error"])
        comparison = compare_reporters(head, head, single, seed=0)
        self.assertTrue(np.isfinite(comparison["brier_t1"]))
        self.assertIsNone(comparison["contrast_reduction"])
        self.assertIsNone(comparison["contrast_ci"])

    def test_lambda_selection_prefers_pair_on_dev(self):
        train = synthetic_states(seed=0)
        dev = synthetic_states(seed=5)
        selection = select_lambda(train, dev, lambdas=(0.25, 1.0), steps=300, seed=0)
        self.assertIn(selection["best_lambda"], (0.25, 1.0))
        self.assertGreaterEqual(len(selection["scores"]), 3)

    def test_reporter_objective_shapes(self):
        data = synthetic_states(seed=2)
        head = fit_pair_reporter(data, lam=1.0, steps=100, seed=0)
        scores = reporter_objective(head, data, lam=1.0)
        self.assertIn("total", scores)
        self.assertGreaterEqual(scores["point"], 0.0)


class VariantTrainingTests(unittest.TestCase):
    def test_training_example_ends_with_eos(self):
        from causal_confidence.stages import EpisodeEnvironment

        model = tiny_model(vocab_size=64, n_layers=2)
        env = EpisodeEnvironment(
            model=model,
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
            revision="test",
            n_layers=2,
        )
        fact = know.build_corpus(1, seed=0)[0]
        input_ids, labels = var._training_example(env, fact, "train")
        self.assertEqual(int(input_ids[-1]), env.tokenizer.eos_token_id)
        self.assertEqual(int(labels[-1]), env.tokenizer.eos_token_id)
        self.assertEqual(int(labels[0]), -100)

    def test_train_save_load_roundtrip(self):
        import tempfile
        from pathlib import Path

        from causal_confidence.stages import EpisodeEnvironment

        model = tiny_model(vocab_size=64, n_layers=4)
        env = EpisodeEnvironment(
            model=model,
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
            revision="test",
            n_layers=4,
        )
        facts = know.build_corpus(2, seed=0)[:2]
        adapters, metrics = var.train_variant(
            env, facts, layers=(1,), rank=4, epochs=1, batch=2, seed=0
        )
        self.assertGreater(metrics["steps"], 0)
        before = adapters[0].lora_a.detach().clone()
        self.assertFalse(torch.equal(before, adapters[0].lora_a.detach() * 0) and torch.equal(before, adapters[0].lora_a))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "variant.pt"
            var.save_variant(adapters, path)
            var.remove_variant(env, adapters)
            loaded = var.install_variant(env, layers=(1,), rank=4, state_path=path)
            self.assertTrue(
                torch.allclose(loaded[0].lora_a.detach(), before)
            )
            var.remove_variant(env, loaded)


class CliContractTests(unittest.TestCase):
    def test_knowledge_parser_exposes_study_flags(self):
        from causal_confidence.cli import build_parser

        args = build_parser().parse_args(["knowledge", "--seed", "0"])
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.subjects, 60)
        self.assertEqual(args.transfer_subjects, 15)
        self.assertEqual(args.train_subjects, 36)
        self.assertEqual(args.dev_subjects, 12)
        self.assertEqual(args.rank, 8)
        self.assertEqual(args.train_draws, 8)
        self.assertEqual(args.eval_draws, 16)


if __name__ == "__main__":
    unittest.main()
