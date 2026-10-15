"""Experiment 1 of the retrieval-monitoring plan: input-only text baselines.

This module answers one question with saved data:

    Can a predictor that sees only the question, and (for the after-answer
    regime) the candidate answer, explain the saved probe advantage?

Two regimes are kept separate, as the review requires:

- ``question``: the prospective regime.  The predictor sees the question text,
  the relation, and the popularity.  It does not see any candidate answer.
  This is the correct baseline for a confidence report that is elicited
  before the answer exists.
- ``question_candidate``: the after-answer regime.  The predictor also sees the
  candidate answer text.  This is the correct baseline for the saved Track B
  probe, which scores an answer that already exists.

The models are trained on the frozen train split and scored on the frozen test
split.  This mirrors the saved Section 3.3 protocol.  The saved calibrated
probe scores (``cal_H_512*``, ``cal_P*``, ``cal_V*``) are read from the test
predictions parquet and compared on the same rows.

Usage:

    python scripts/run_text_baselines.py --run runs/section33_confirmatory \
        --track B --track A
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .difficulty import within_question_auroc

SEED = 0


@dataclass
class BaselineReport:
    track: str
    mode: str
    n_train_rows: int
    n_train_questions: int
    n_test_rows: int
    n_test_questions: int
    overall_auroc: float
    within_question_auroc: float | None
    eligible_questions: int
    saved: dict[str, float]
    delta_vs_hidden: float
    delta_ci: tuple[float, float]
    within_delta_vs_hidden: float | None

    def as_dict(self) -> dict:
        return {
            "track": self.track,
            "mode": self.mode,
            "n_train_rows": self.n_train_rows,
            "n_train_questions": self.n_train_questions,
            "n_test_rows": self.n_test_rows,
            "n_test_questions": self.n_test_questions,
            "overall_auroc": self.overall_auroc,
            "within_question_auroc": self.within_question_auroc,
            "eligible_questions": self.eligible_questions,
            "saved": self.saved,
            "delta_vs_hidden": self.delta_vs_hidden,
            "delta_ci": list(self.delta_ci),
            "within_delta_vs_hidden": self.within_delta_vs_hidden,
        }


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_answers(run_dir: Path, track: str) -> pd.DataFrame:
    """Load all answer rows for one track, with duplicates removed.

    The per-shard files and the combined file can overlap.  Rows are keyed by
    question, answer index, generation seed, and candidate key.  The first
    occurrence wins.
    """

    pattern = f"track{track}_answers__*.jsonl"
    files = sorted((run_dir / "labels").glob(pattern))
    if not files:
        raise FileNotFoundError(f"no answer files match {pattern} in {run_dir}")
    seen: set[tuple] = set()
    rows: list[dict] = []
    for path in files:
        for row in _read_jsonl(path):
            key = (
                row.get("question_id"),
                row.get("answer_index"),
                row.get("generation_seed"),
                row.get("candidate_key"),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    frame = pd.DataFrame(rows)
    keep = [
        "question_id",
        "subject_id",
        "split",
        "relation",
        "popularity",
        "stratum",
        "candidate_text",
        "label",
        "multiplicity",
        "track",
    ]
    frame = frame[[column for column in keep if column in frame.columns]].copy()
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame["popularity"] = pd.to_numeric(frame["popularity"], errors="coerce")
    frame["multiplicity"] = pd.to_numeric(frame.get("multiplicity"), errors="coerce").fillna(1)
    return frame


def load_questions(run_dir: Path) -> pd.DataFrame:
    """Question text and metadata for every cohort row."""

    manifest = run_dir / "manifests" / "cohort.parquet"
    if not manifest.exists():
        raise FileNotFoundError(f"missing cohort manifest: {manifest}")
    frame = pd.read_parquet(manifest)
    if "question" not in frame.columns:
        # Fall back to the source table when the cohort lacks question text.
        source = run_dir / "source" / "popqa_source.parquet"
        source_frame = pd.read_parquet(source)
        frame = frame.merge(
            source_frame[["question_id", "question"]]
            if "question_id" in source_frame.columns
            else source_frame[["id", "question"]].rename(columns={"id": "question_id"}),
            on="question_id",
            how="left",
        )
    keep = ["question_id", "question", "relation", "popularity", "stratum", "split", "subject_id"]
    return frame[[column for column in keep if column in frame.columns]].copy()


def _expand_multiplicity(frame: pd.DataFrame, cap: int = 8) -> pd.DataFrame:
    """Expand candidate rows by their multiplicity, capped.

    The saved Track B evaluation expands candidates by how often they appeared
    among the eight draws.  The text baseline uses the same weighting so the
    comparison is not an artefact of the aggregate.
    """

    if "multiplicity" not in frame.columns:
        return frame
    weights = frame["multiplicity"].fillna(1).clip(lower=1, upper=cap).astype(int)
    if int(weights.sum()) == len(frame):
        return frame
    return frame.loc[frame.index.repeat(weights)].reset_index(drop=True)


def _dense_metadata(frame: pd.DataFrame) -> np.ndarray:
    relation = frame[["relation"]].fillna("missing")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    one_hot = encoder.fit_transform(relation)
    popularity = np.log10(frame["popularity"].fillna(0).to_numpy(dtype=float) + 1.0).reshape(-1, 1)
    return np.hstack([one_hot, popularity])


def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    mode: str,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Fit text and metadata features on the train rows, transform both sets."""

    if mode not in {"question", "question_candidate"}:
        raise ValueError(f"unknown mode: {mode}")
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=200_000,
        strip_accents="unicode",
    )
    q_train = word.fit_transform(train["question"].fillna(""))
    q_test = word.transform(test["question"].fillna(""))
    blocks_train = [q_train]
    blocks_test = [q_test]
    if mode == "question_candidate":
        char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=3,
            sublinear_tf=True,
            max_features=200_000,
        )
        c_train = char.fit_transform(train["candidate_text"].fillna(""))
        c_test = char.transform(test["candidate_text"].fillna(""))
        blocks_train.append(c_train)
        blocks_test.append(c_test)
    meta_scaler = StandardScaler()
    m_train = meta_scaler.fit_transform(_dense_metadata(train))
    m_test = meta_scaler.transform(_dense_metadata(test))
    blocks_train.append(sparse.csr_matrix(m_train))
    blocks_test.append(sparse.csr_matrix(m_test))
    return (
        sparse.hstack(blocks_train).tocsr(),
        sparse.hstack(blocks_test).tocsr(),
    )


def fit_text_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    mode: str,
    c_value: float = 1.0,
) -> np.ndarray:
    train_expanded = _expand_multiplicity(train)
    x_train, x_test = build_features(train_expanded, test, mode)
    model = LogisticRegression(C=c_value, max_iter=3000, random_state=SEED)
    model.fit(x_train, train_expanded["label"].to_numpy(dtype=int))
    return model.predict_proba(x_test)[:, 1]


def _subject_bootstrap_delta(
    frame: pd.DataFrame,
    scores: np.ndarray,
    reference: np.ndarray,
    n_boot: int = 200,
    seed: int = SEED,
) -> tuple[float, tuple[float, float]]:
    """Subject-clustered bootstrap of the overall AUROC difference."""

    labels = frame["label"].to_numpy(dtype=int)
    subjects = frame["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    index = {subject: np.where(subjects == subject)[0] for subject in unique}
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[subject] for subject in drawn])
        y = labels[rows]
        if len(np.unique(y)) < 2:
            continue
        deltas.append(
            roc_auc_score(y, scores[rows]) - roc_auc_score(y, reference[rows])
        )
    if not deltas:
        return float("nan"), (float("nan"), float("nan"))
    array = np.asarray(deltas, dtype=float)
    return float(np.mean(array)), (
        float(np.percentile(array, 2.5)),
        float(np.percentile(array, 97.5)),
    )


def _question_bootstrap_delta(
    frame: pd.DataFrame,
    scores: np.ndarray,
    reference: np.ndarray,
    n_boot: int = 500,
    seed: int = SEED,
) -> tuple[float | None, tuple[float, float] | None]:
    """Bootstrap over mixed-outcome questions of the within-question delta."""

    labels = frame["label"].to_numpy(dtype=int)
    questions = frame["question_id"].to_numpy()
    per_question: list[tuple[float, float]] = []
    for question in np.unique(questions):
        mask = questions == question
        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue
        first = within_question_auroc(scores[mask], y, np.zeros(mask.sum()))[
            "within_question_auroc"
        ]
        second = within_question_auroc(reference[mask], y, np.zeros(mask.sum()))[
            "within_question_auroc"
        ]
        if first is not None and second is not None:
            per_question.append((first, second))
    if not per_question:
        return None, None
    values = np.asarray(per_question, dtype=float)
    observed = float(np.mean(values[:, 0] - values[:, 1]))
    rng = np.random.default_rng(seed)
    differences = values[:, 0] - values[:, 1]
    draws = [
        float(np.mean(rng.choice(differences, size=len(differences), replace=True)))
        for _ in range(n_boot)
    ]
    array = np.asarray(draws, dtype=float)
    return observed, (float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5)))


def evaluate_baseline(
    track: str,
    mode: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    saved_columns: dict[str, str],
    hidden_column: str,
) -> tuple[BaselineReport, pd.DataFrame]:
    """Train one baseline and report the comparison with the saved scores."""

    train_split = train[train["split"] == "train"].copy()
    test_split = test.copy()
    scores = fit_text_baseline(train_split, test_split, mode)
    labels = test_split["label"].to_numpy(dtype=int)
    overall = float(roc_auc_score(labels, scores))
    within = within_question_auroc(
        scores, labels, test_split["question_id"].to_numpy()
    )
    saved = {
        name: float(roc_auc_score(labels, test_split[column].to_numpy(dtype=float)))
        for name, column in saved_columns.items()
    }
    delta, interval = _subject_bootstrap_delta(
        test_split, scores, test_split[hidden_column].to_numpy(dtype=float)
    )
    within_delta, _ = _question_bootstrap_delta(
        test_split, scores, test_split[hidden_column].to_numpy(dtype=float)
    )
    report = BaselineReport(
        track=track,
        mode=mode,
        n_train_rows=int(len(train_split)),
        n_train_questions=int(train_split["question_id"].nunique()),
        n_test_rows=int(len(test_split)),
        n_test_questions=int(test_split["question_id"].nunique()),
        overall_auroc=overall,
        within_question_auroc=within["within_question_auroc"],
        eligible_questions=within["eligible_questions"],
        saved=saved,
        delta_vs_hidden=delta,
        delta_ci=interval,
        within_delta_vs_hidden=within_delta,
    )
    score_frame = test_split[
        ["question_id", "subject_id", "label", "relation", "popularity", "stratum"]
        + [column for column in saved_columns.values() if column in test_split.columns]
    ].copy()
    score_frame["track"] = track
    score_frame["mode"] = mode
    score_frame["text_score"] = scores
    return report, score_frame


def prepare(run_dir: Path, track: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join answer rows with question text and the saved test predictions."""

    answers = load_answers(run_dir, track)
    questions = load_questions(run_dir)
    question_map = questions[["question_id", "question"]].drop_duplicates("question_id")
    answers = answers.merge(question_map, on="question_id", how="left")
    predictions = pd.read_parquet(
        run_dir / "predictions" / f"test_predictions__qwen__qwen2.5_7b_instruct__track{track}.parquet"
    )
    predictions = predictions.merge(question_map, on="question_id", how="left")
    if "label" not in predictions.columns:
        raise ValueError("test predictions have no label column")
    return answers, predictions


def run_track(
    run_dir: Path, track: str, modes: tuple[str, ...]
) -> tuple[list[BaselineReport], pd.DataFrame]:
    answers, predictions = prepare(run_dir, track)
    saved_columns = {
        "saved_H": "cal_H_512_after" if track == "B" else "cal_H_512",
        "saved_P": "cal_P_after" if track == "B" else "cal_P",
        "saved_V": "cal_V_after" if track == "B" else "cal_V",
    }
    reports: list[BaselineReport] = []
    score_frames: list[pd.DataFrame] = []
    for mode in modes:
        report, scores = evaluate_baseline(
            track=track,
            mode=mode,
            train=answers,
            test=predictions,
            saved_columns=saved_columns,
            hidden_column=saved_columns["saved_H"],
        )
        reports.append(report)
        score_frames.append(scores)
    return reports, pd.concat(score_frames, ignore_index=True)


def save_reports(
    run_dir: Path, reports: list[BaselineReport], scores: pd.DataFrame | None = None
) -> Path:
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "text_baselines.json"
    payload = {"reports": [report.as_dict() for report in reports]}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    if scores is not None:
        scores.to_parquet(out_dir / "text_baselines_scores.parquet", index=False)
    return path
