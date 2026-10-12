"""GPU runner for the S3 pilot.

Two entry paths:

- ``run_pilot`` uses a cohort with two questions per subject (the original
  design).
- ``run_pilot_from_source`` builds the pilot pairs from the raw Wikidata
  source, so the one-question-per-subject confirmatory cohort is never needed
  for the pilot.  Subjects of the frozen cohort are excluded.

The analysis functions in ``pilot`` are CPU-testable; this module needs a real
model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from section33.data.cohort import normalize_columns, parse_reference_answers
from section33.models.loader import load_model_and_tokenizer

from .collector import CollectorSettings, RetainedCollector, capture_subject_states
from .interventions import fit_relation_map
from .pairs import load_pairs_csv
from .pilot import (
    build_pairs_from_source,
    load_rows,
    save_report,
    select_subject_pairs,
    selectivity_report,
)
from .text_baselines import load_questions


def _first_token_embedding(model, tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer(str(text), add_special_tokens=False)["input_ids"][:1]
    if not ids:
        ids = [tokenizer.eos_token_id]
    embedding = model.get_input_embeddings()
    with torch.no_grad():
        return embedding(torch.tensor(ids, device=next(model.parameters()).device))[0]


def _fit_map(
    loaded,
    subjects: list[dict],
    relation: str,
    layer_index: int,
    rank: int,
):
    states = capture_subject_states(loaded, subjects, layer_index)
    state_tensor = torch.stack([states[str(subject["subject_id"])] for subject in subjects])
    answers = torch.stack(
        [
            _first_token_embedding(loaded.model, loaded.tokenizer, subject["answer"])
            for subject in subjects
        ]
    )
    return fit_relation_map(state_tensor, answers, relation, rank=rank)


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
            {
                "subject_id": str(row.subject_id),
                "name": str(row.subject_name),
                "answer": row.reference_answers[0]
                if isinstance(row.reference_answers, (list, tuple))
                else row.reference_answers,
            }
            for row in block.itertuples()
        ]
        maps[relation] = _fit_map(loaded, subjects, relation, layer_index, rank)
    return maps


def load_normalized_source(source_csv: Path) -> pd.DataFrame:
    source = normalize_columns(pd.read_csv(source_csv))
    source["reference_answers"] = source["reference_answers"].map(parse_reference_answers)
    return source


def _cohort_subject_ids(run_dir: Path | None) -> set[str]:
    if run_dir is None:
        return set()
    path = run_dir / "manifests" / "cohort.parquet"
    if not path.exists():
        return set()
    cohort = pd.read_parquet(path)
    return set(cohort["subject_id"].astype(str))


def fit_maps_from_source(
    loaded,
    source: pd.DataFrame,
    relations: tuple[str, ...],
    layer_index: int,
    rank: int = 2,
    per_relation: int = 200,
    exclude: set[str] | None = None,
):
    exclude = exclude or set()
    maps = {}
    for relation in relations:
        block = source[source["relation"] == relation].copy()
        block["subject_id"] = block["subject_id"].astype(str)
        block = block[~block["subject_id"].isin(exclude)]
        block = block.drop_duplicates("subject_id").head(per_relation)
        if block.empty:
            raise ValueError(f"no map-fitting subjects for relation {relation}")
        subjects = [
            {
                "subject_id": str(row.subject_id),
                "name": str(row.subject_name),
                "answer": row.reference_answers[0]
                if isinstance(row.reference_answers, (list, tuple))
                else row.reference_answers,
            }
            for row in block.itertuples()
        ]
        maps[relation] = _fit_map(loaded, subjects, relation, layer_index, rank)
    return maps


def _collect_and_report(
    loaded,
    pairs: list[dict],
    relation_maps: dict,
    *,
    layer_index: int,
    alpha: float,
    out_dir: Path,
    model_name: str,
) -> dict:
    pilot_subjects = [
        {"subject_id": pair["subject_id"], "name": pair["name"]} for pair in pairs
    ]
    states = capture_subject_states(loaded, pilot_subjects, layer_index)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "pilot_rows.jsonl"
    collector = RetainedCollector(
        loaded,
        CollectorSettings(layer_index=layer_index, alpha=alpha),
        relation_maps,
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
    out_dir = out_dir or (run_dir / "retrieval_monitoring")
    return _collect_and_report(
        loaded,
        pairs,
        maps,
        layer_index=layer_index,
        alpha=alpha,
        out_dir=out_dir,
        model_name=model_name,
    )


def run_pilot_from_source(
    source_csv: Path,
    model_name: str,
    *,
    run_dir: Path | None = None,
    pairs_csv: Path | None = None,
    n_subjects: int = 40,
    relations: tuple[str, str] = ("father", "occupation"),
    layer_index: int | None = None,
    alpha: float = 0.5,
    rank: int = 2,
    dtype: str = "float16",
    map_subjects: int = 200,
    out_dir: Path | None = None,
) -> dict:
    """Build pilot pairs and run the selectivity pilot.

    ``pairs_csv`` comes from ``retrieval_monitoring.pairs`` and holds subjects
    that have both relations.  Without it, the runner tries to find pairs in
    the raw source, which rarely works because each relation is fetched
    independently.
    """

    source = load_normalized_source(source_csv)
    exclude = _cohort_subject_ids(run_dir)
    if pairs_csv is not None:
        pairs = load_pairs_csv(pairs_csv, relations)
        if len(pairs) < n_subjects:
            raise ValueError(
                f"pairs CSV holds {len(pairs)} complete pairs; need {n_subjects}. "
                "Fetch more with `python -m retrieval_monitoring.cli fetch-pairs`."
            )
        pairs = pairs[:n_subjects]
    else:
        pairs = build_pairs_from_source(source, relations, n_subjects, exclude)
    pilot_ids = {pair["subject_id"] for pair in pairs}
    loaded = load_model_and_tokenizer(model_name, dtype=dtype)
    if layer_index is None:
        layer_index = max(0, int(loaded.model.config.num_hidden_layers) // 2 - 1)
    maps = fit_maps_from_source(
        loaded,
        source,
        relations,
        layer_index,
        rank=rank,
        per_relation=map_subjects,
        exclude=pilot_ids | exclude,
    )
    out_dir = out_dir or (Path(run_dir) / "retrieval_monitoring" if run_dir else Path("retrieval_monitoring"))
    return _collect_and_report(
        loaded,
        pairs,
        maps,
        layer_index=layer_index,
        alpha=alpha,
        out_dir=out_dir,
        model_name=model_name,
    )
