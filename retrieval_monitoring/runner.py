"""GPU runner for the S3 pilot.

Fits relation maps on training subjects, collects the pilot cells, and writes
the selectivity report.  This path needs a real model; the analysis functions
in ``pilot`` are CPU-testable.
"""

from __future__ import annotations

from pathlib import Path

import torch

from section33.models.loader import load_model_and_tokenizer

from .collector import CollectorSettings, RetainedCollector, capture_subject_states
from .interventions import fit_relation_map
from .pilot import load_rows, save_report, select_subject_pairs, selectivity_report
from .text_baselines import load_questions


def _first_token_embedding(model, tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer(str(text), add_special_tokens=False)["input_ids"][:1]
    if not ids:
        ids = [tokenizer.eos_token_id]
    embedding = model.get_input_embeddings()
    with torch.no_grad():
        return embedding(torch.tensor(ids, device=next(model.parameters()).device))[0]


def fit_maps_for_cohort(
    loaded,
    cohort,
    relations: tuple[str, ...],
    layer_index: int,
    rank: int = 2,
    max_subjects: int = 200,
):
    """Fit one relation map per relation from training subjects."""

    train = cohort[cohort["split"] == "train"]
    maps = {}
    for relation in relations:
        block = train[train["relation"] == relation].head(max_subjects)
        if block.empty:
            raise ValueError(f"no training subjects for relation {relation}")
        subjects = [
            {"subject_id": str(row.subject_id), "name": str(row.subject_name)}
            for row in block.itertuples()
        ]
        states = capture_subject_states(loaded, subjects, layer_index)
        state_tensor = torch.stack([states[str(row.subject_id)] for row in block.itertuples()])
        answers = torch.stack(
            [
                _first_token_embedding(
                    loaded.model,
                    loaded.tokenizer,
                    row.reference_answers[0]
                    if isinstance(row.reference_answers, (list, tuple))
                    else row.reference_answers,
                )
                for row in block.itertuples()
            ]
        )
        maps[relation] = fit_relation_map(state_tensor, answers, relation, rank=rank)
    return maps


def run_pilot(
    run_dir: Path,
    model_name: str,
    *,
    n_subjects: int = 40,
    relations: tuple[str, str] = ("father", "occupation"),
    layer_index: int | None = None,
    alpha: float = 0.5,
    rank: int = 2,
    dtype: str = "float16",
    out_dir: Path | None = None,
) -> dict:
    cohort = load_questions(run_dir)
    pairs = select_subject_pairs(cohort, n_subjects, relations, split="dev")
    loaded = load_model_and_tokenizer(model_name, dtype=dtype)
    if layer_index is None:
        layer_index = max(0, int(loaded.model.config.num_hidden_layers) // 2 - 1)
    maps = fit_maps_for_cohort(loaded, cohort, relations, layer_index, rank=rank)
    pilot_subjects = [
        {"subject_id": pair["subject_id"], "name": pair["name"]} for pair in pairs
    ]
    states = capture_subject_states(loaded, pilot_subjects, layer_index)
    out_dir = out_dir or (run_dir / "retrieval_monitoring")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "pilot_rows.jsonl"
    collector = RetainedCollector(
        loaded,
        CollectorSettings(layer_index=layer_index, alpha=alpha),
        maps,
    )
    collection = collector.collect(pairs, states, rows_path)
    rows = load_rows(rows_path)
    report = selectivity_report(rows)
    report["collection"] = collection
    report["model"] = model_name
    report["layer"] = layer_index
    report["alpha"] = alpha
    save_report(report, out_dir / "selectivity_report.json")
    return report
