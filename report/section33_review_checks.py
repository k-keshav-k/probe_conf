"""Post-repair reproduction checks for the Section 3.3 implementation review.

Run with:

    .venv/bin/python report/section33_review_checks.py

The pre-repair evidence is kept in ``report/SECTION33_REVIEW_EVIDENCE.json``.
This script exercises the same interfaces against the corrected code and prints
a JSON report (the post-repair evidence).
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from section33.analysis.interventions import flatten_sharpen
from section33.analysis.power import (
    development_variance,
    paired_development_distribution,
    simulate_paired_decision,
)
from section33.datasets import load_split, metadata_features
from section33.probes.pipeline import assemble_blocks, select_columns, standard_specs

root = Path(__file__).resolve().parents[1]
run = root / "runs/section33_v1"
out = {}
specs = standard_specs()

# F1: the probability blocks must select the probability columns.
for track, name in [("A", "P"), ("B", "P_after")]:
    path = sorted((run / "features").glob(f"track{track}_features*.jsonl"))[0]
    with path.open() as handle:
        row = json.loads(next(handle))
    frame = pd.DataFrame([row])
    out[f"probability_{track}"] = {
        "available_columns": [c for c in frame if c.startswith("f__prob__")],
        "selected_columns": select_columns(frame, specs[name].blocks[0]),
        "baseline_block_dimensions": {
            k: v.shape[1]
            for k, v in assemble_blocks(frame, specs["B" if track == "A" else "B_after"]).items()
        },
    }

# F2: generated-answer length must not enter the Track A baseline.
frame = pd.DataFrame(
    {"track": ["A", "A"], "content_length": [1, 7], "relation": ["father", "father"]}
)
metadata = metadata_features(frame, track="A").add_prefix("f__meta__")
out["trackA_answer_length_in_baseline"] = {
    "selected_columns": select_columns(metadata, specs["B"].blocks[0]),
    "values": assemble_blocks(metadata, specs["B"])["meta"].tolist(),
}

# F5: one-answer subjects now produce a pooled bootstrap distribution.
rng = np.random.default_rng(0)
n = 40
single = pd.DataFrame(
    {
        "subject_id": [f"s{i}" for i in range(n)],
        "question_id": [f"q{i}" for i in range(n)],
        "label": rng.integers(0, 2, n),
        "a": rng.random(n),
        "b": rng.random(n),
        "relation": "father",
        "stratum": "0",
    }
)
distribution = paired_development_distribution(single, "a", "b", n_boot=50, seed=0)
variance = development_variance(distribution)
out["trackA_power_distribution"] = {
    "n_subjects": distribution["n_subjects"],
    "n_bootstrap_draws": len(distribution["delta_auc"]),
    "variance": {
        key: (None if not np.isfinite(value) else round(value, 6))
        for key, value in variance.items()
    },
}
out["power_large_sample"] = [
    simulate_paired_decision(
        n_subjects,
        0.04,
        0.01,
        se_auc=0.2 / np.sqrt(n_subjects),
        se_brier=0.05 / np.sqrt(n_subjects),
        n_sim=10_000,
        seed=123,
        calibration_ok_probability=1,
    )
    for n_subjects in (100, 1_000_000)
]

# F3: distinct token sequences with the same decoded text must not merge.
with tempfile.TemporaryDirectory() as tmp:
    paths = {name: Path(tmp) / name for name in ("labels", "features")}
    for path in paths.values():
        path.mkdir()
    common = dict(subject_id="s", question_id="q", split="test", track="B")
    labels = [
        {**common, "answer_index": i, "candidate_text": "same answer", "label": 1, "multiplicity": 4}
        for i in range(2)
    ]
    features = [
        {
            **common,
            "answer_index": i,
            "candidate_token_ids": [i + 10],
            "position": "answer_last_content",
            "multiplicity": 4,
            "content_length": 1,
            "f__hiddenproj__post__answer_last_content__l0__k0": float(i),
        }
        for i in range(2)
    ]
    for path, rows in [
        (paths["labels"] / "trackB_answers__stub.jsonl", labels),
        (paths["features"] / "trackB_features__stub.jsonl", features),
    ]:
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    joined = load_split(paths, "stub", "B", expected_samples=8)
    out["candidate_join_reproduction"] = {
        "input_samples": 8,
        "joined_samples": int(joined["multiplicity"].sum()),
        "input_candidates": 2,
        "joined_candidates": len(joined),
    }

# F10: the intervention must be invariant to function-preserving rescaling.
s = torch.tensor([1.0, 1.0])
w = torch.diag(torch.tensor([1.0, 10.0]))
scale = torch.tensor([10.0, 1.0])
normal = flatten_sharpen(s, w, exponent=0.5)
rescaled = flatten_sharpen(s * scale, w / scale, exponent=0.5)
out["flatten_sharpen_invariance"] = {
    "original_output": (w @ s).tolist(),
    "equivalent_output": ((w / scale) @ (s * scale)).tolist(),
    "intervened_output": normal.o_new.tolist(),
    "equivalent_intervened_output": rescaled.o_new.tolist(),
    "invariant": bool(torch.allclose(normal.o_new, rescaled.o_new, atol=1e-5)),
}

print(json.dumps(out, indent=2))
