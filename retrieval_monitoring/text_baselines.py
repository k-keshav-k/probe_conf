"""S0: input-only text baselines on the frozen cohort (repaired).

The earlier loader pooled every matching label file, so it used 2,300 training
subjects from a 2,000-subject manifest, mixed collections, and inconsistent
draw counts.  This module rebuilds both tracks from the frozen cohort manifest
and the canonical (unsharded) collection, validates per-question draw counts,
and saves both tracks with provenance and intervals.

Regimes
-------
- ``question``: prospective.  The predictor sees the question text, relation,
  popularity, and prompt length.  No candidate answer.
- ``question_candidate``: after answer.  The predictor also sees the candidate
  answer text.

Both are TF-IDF plus metadata logistic baselines with the regularization
strength selected on the development split.  This is a tested baseline, not the
strongest possible input-only observer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 0
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0)


@dataclass
class BaselineReport:
    track: str
    mode: str
    n_train_questions: int
    n_train_rows: int
    n_train_subjects: int
    n_dev_questions: int
    n_dev_rows: int
    n_test_questions: int
    n_test_rows: int
    n_test_subjects: int
    selected_c: float
    overall_auroc: float
    within_question_auroc: float | None
    eligible_questions: int
    saved: dict[str, float]
    observed_delta_vs_hidden: float
    delta_ci: tuple[float, float]
    bootstrap_mean_delta: float
    within_observed_delta: float | None
    within_delta_ci: tuple[float, float] | None
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "track": self.track,
            "mode": self.mode,
            "n_train_questions": self.n_train_questions,
            "n_train_rows": self.n_train_rows,
            "n_train_subjects": self.n_train_subjects,
            "n_dev_questions": self.n_dev_questions,
            "n_dev_rows": self.n_dev_rows,
            "n_test_questions": self.n_test_questions,
            "n_test_rows": self.n_test_rows,
            "n_test_subjects": self.n_test_subjects,
            "selected_c": self.selected_c,
            "overall_auroc": self.overall_auroc,
            "within_question_auroc": self.within_question_auroc,
            "eligible_questions": self.eligible_questions,
            "saved": self.saved,
            "observed_delta_vs_hidden": self.observed_delta_vs_hidden,
            "delta_ci": list(self.delta_ci),
            "bootstrap_mean_delta": self.bootstrap_mean_delta,
            "within_observed_delta": self.within_observed_delta,
            "within_delta_ci": list(self.within_delta_ci) if self.within_delta_ci else None,
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def canonical_answer_files(run_dir: Path, track: str) -> list[Path]:
    """Canonical answer files: the unsharded collection when it exists."""

    pattern = f"track{track}_answers__*.jsonl"
    files = sorted((run_dir / "labels").glob(pattern))
    unsharded = [path for path in files if "__shard" not in path.name]
    selected = unsharded or files
    if not selected:
        raise FileNotFoundError(f"no answer files match {pattern} in {run_dir}")
    return selected


def load_questions(run_dir: Path) -> pd.DataFrame:
    """Question text, relation, popularity, stratum, split, and subject."""

    manifest = run_dir / "manifests" / "cohort.parquet"
    if not manifest.exists():
        raise FileNotFoundError(f"missing cohort manifest: {manifest}")
    frame = pd.read_parquet(manifest)
    if "question" not in frame.columns:
        source = run_dir / "source" / "popqa_source.parquet"
        if not source.exists():
            raise ValueError("cohort has no question text and no source table")
        source_frame = pd.read_parquet(source)
        if "question_id" in source_frame.columns:
            merge_frame = source_frame[["question_id", "question"]]
        else:
            merge_frame = source_frame[["id", "question"]].rename(columns={"id": "question_id"})
        frame = frame.merge(merge_frame, on="question_id", how="left")
    keep = [
        "question_id",
        "question",
        "relation",
        "popularity",
        "stratum",
        "split",
        "subject_id",
        "subject_name",
        "reference_answers",
    ]
    return frame[[column for column in keep if column in frame.columns]].copy()


def load_answers(
    run_dir: Path,
    track: str,
    cohort: pd.DataFrame,
    expected_draws: int = 8,
) -> pd.DataFrame:
    """Load the canonical collection, filtered to the frozen cohort.

    Validation:
    - every question belongs to the cohort and its split matches the manifest;
    - Track A has exactly one row per question;
    - Track B has exactly ``expected_draws`` total multiplicity per question;
    - repeated candidate text is preserved, not deduplicated.
    """

    rows: list[dict] = []
    for path in canonical_answer_files(run_dir, track):
        rows.extend(_read_jsonl(path))
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no answer rows for track {track}")

    cohort_ids = set(cohort["question_id"].astype(str))
    frame["question_id"] = frame["question_id"].astype(str)
    frame = frame[frame["question_id"].isin(cohort_ids)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"track {track} has no rows inside the frozen cohort")

    split_map = cohort.set_index("question_id")["split"].astype(str)
    frame["split"] = frame["question_id"].map(split_map)
    if "multiplicity" not in frame.columns:
        frame["multiplicity"] = 1
    frame["multiplicity"] = pd.to_numeric(frame["multiplicity"], errors="coerce").fillna(1).astype(int)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")

    if track.upper() == "A":
        counts = frame.groupby("question_id").size()
        bad = counts[counts != 1]
        if len(bad):
            raise ValueError(
                f"Track A draw counts are not one per question: "
                f"{len(bad)} questions differ; examples {bad.head(5).to_dict()}"
            )
    else:
        totals = frame.groupby("question_id")["multiplicity"].sum()
        bad = totals[totals != expected_draws]
        if len(bad):
            raise ValueError(
                f"Track B draw counts are not {expected_draws} per question: "
                f"{len(bad)} questions differ; examples {bad.head(5).to_dict()}"
            )
        key = frame.get("candidate_key")
        if key is None or key.isna().any():
            frame["candidate_key"] = (
                frame["question_id"].astype(str)
                + "::"
                + frame.get("answer_index", pd.Series(0, index=frame.index)).astype(str)
                + "::"
                + frame.get("candidate_text", pd.Series("", index=frame.index)).astype(str)
            )
        duplicated = frame.duplicated(subset=["question_id", "candidate_key"], keep=False)
        if duplicated.any():
            # Identical records from overlapping files are allowed; conflicting
            # records with the same key are not.
            conflicts = frame[duplicated].drop_duplicates()
            if len(conflicts) != len(frame[duplicated].drop_duplicates(subset=["question_id", "candidate_key"])):
                raise ValueError("conflicting duplicate candidate rows in the collection")
            frame = frame.drop_duplicates(subset=["question_id", "candidate_key"], keep="first")

    frame["popularity"] = pd.to_numeric(frame["popularity"], errors="coerce")
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Features and fitting
# ---------------------------------------------------------------------------
def _dense_metadata(frame: pd.DataFrame) -> np.ndarray:
    relation = frame[["relation"]].fillna("missing")
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    one_hot = encoder.fit_transform(relation)
    popularity = np.log10(frame["popularity"].fillna(0).to_numpy(dtype=float) + 1.0).reshape(-1, 1)
    question = frame.get("question", pd.Series("", index=frame.index)).fillna("")
    length = question.str.len().to_numpy(dtype=float).reshape(-1, 1)
    return np.hstack([one_hot, popularity, length])


def build_features(train: pd.DataFrame, test: pd.DataFrame, mode: str):
    if mode not in {"question", "question_candidate"}:
        raise ValueError(f"unknown mode: {mode}")
    word = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000,
        strip_accents="unicode",
    )
    blocks_train = [word.fit_transform(train["question"].fillna(""))]
    blocks_test = [word.transform(test["question"].fillna(""))]
    if mode == "question_candidate":
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True,
            max_features=200_000,
        )
        blocks_train.append(char.fit_transform(train["candidate_text"].fillna("")))
        blocks_test.append(char.transform(test["candidate_text"].fillna("")))
    scaler = StandardScaler()
    blocks_train.append(sparse.csr_matrix(scaler.fit_transform(_dense_metadata(train))))
    blocks_test.append(sparse.csr_matrix(scaler.transform(_dense_metadata(test))))
    return sparse.hstack(blocks_train).tocsr(), sparse.hstack(blocks_test).tocsr()


def fit_text_baseline(
    train: pd.DataFrame, test: pd.DataFrame, mode: str, c_value: float
) -> np.ndarray:
    x_train, x_test = build_features(train, test, mode)
    model = LogisticRegression(C=c_value, max_iter=3000, random_state=SEED)
    model.fit(x_train, train["label"].to_numpy(dtype=int))
    return model.predict_proba(x_test)[:, 1]


def _expand_multiplicity(frame: pd.DataFrame, cap: int = 8) -> pd.DataFrame:
    """Repeat candidate rows by their sampling multiplicity (Track B)."""

    if "multiplicity" not in frame.columns:
        return frame
    weights = frame["multiplicity"].fillna(1).clip(lower=1, upper=cap).astype(int)
    if int(weights.sum()) == len(frame):
        return frame
    return frame.loc[frame.index.repeat(weights)].reset_index(drop=True)


def select_c(train: pd.DataFrame, dev: pd.DataFrame, mode: str) -> tuple[float, dict]:
    """Development-selected regularization strength (AUROC tie-break: Brier)."""

    train_expanded = _expand_multiplicity(train)
    records = {}
    best_c, best_key = C_GRID[0], None
    for c_value in C_GRID:
        scores = fit_text_baseline(train_expanded, dev, mode, c_value)
        labels = dev["label"].to_numpy(dtype=int)
        auroc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else float("nan")
        brier = float(np.mean((scores - labels) ** 2))
        records[c_value] = {"dev_auroc": auroc, "dev_brier": brier}
        key = (-auroc, brier)
        if best_key is None or key < best_key:
            best_key, best_c = key, c_value
    return best_c, records


# ---------------------------------------------------------------------------
# Metrics and intervals
# ---------------------------------------------------------------------------
def within_question_auroc(scores, labels, question_ids) -> dict:
    """Mean within-question AUROC over mixed-outcome questions (ties half)."""

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    question_ids = np.asarray(question_ids)
    values = []
    for question in np.unique(question_ids):
        mask = question_ids == question
        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue
        s = scores[mask]
        positive, negative = s[y == 1], s[y == 0]
        wins = (positive[:, None] > negative[None, :]).sum() + 0.5 * (
            positive[:, None] == negative[None, :]
        ).sum()
        values.append(float(wins / (len(positive) * len(negative))))
    return {
        "within_question_auroc": float(np.mean(values)) if values else None,
        "eligible_questions": len(values),
        "n_questions": int(len(np.unique(question_ids))),
    }


def _subject_bootstrap_delta(
    frame: pd.DataFrame,
    scores: np.ndarray,
    reference: np.ndarray,
    n_boot: int = 500,
    seed: int = SEED,
):
    labels = frame["label"].to_numpy(dtype=int)
    subjects = frame["subject_id"].astype(str).to_numpy()
    unique = np.unique(subjects)
    index = {subject: np.where(subjects == subject)[0] for subject in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        drawn = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[subject] for subject in drawn])
        y = labels[rows]
        if len(np.unique(y)) < 2:
            continue
        draws.append(roc_auc_score(y, scores[rows]) - roc_auc_score(y, reference[rows]))
    observed = roc_auc_score(labels, scores) - roc_auc_score(labels, reference)
    if not draws:
        return float(observed), (float("nan"), float("nan")), float("nan")
    array = np.asarray(draws, dtype=float)
    return (
        float(observed),
        (float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))),
        float(np.mean(array)),
    )


def _question_bootstrap_delta(
    frame: pd.DataFrame,
    scores: np.ndarray,
    reference: np.ndarray,
    n_boot: int = 500,
    seed: int = SEED,
):
    labels = frame["label"].to_numpy(dtype=int)
    questions = frame["question_id"].to_numpy()
    differences = []
    for question in np.unique(questions):
        mask = questions == question
        y = labels[mask]
        if len(np.unique(y)) < 2:
            continue
        first = within_question_auroc(scores[mask], y, np.zeros(mask.sum()))["within_question_auroc"]
        second = within_question_auroc(reference[mask], y, np.zeros(mask.sum()))["within_question_auroc"]
        if first is not None and second is not None:
            differences.append(first - second)
    if not differences:
        return None, None
    differences = np.asarray(differences, dtype=float)
    observed = float(np.mean(differences))
    rng = np.random.default_rng(seed)
    draws = np.asarray(
        [np.mean(rng.choice(differences, size=len(differences), replace=True)) for _ in range(n_boot)],
        dtype=float,
    )
    return observed, (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_baseline(
    track: str,
    mode: str,
    answers: pd.DataFrame,
    test: pd.DataFrame,
    saved_columns: dict[str, str],
    hidden_column: str,
    provenance: dict,
) -> tuple[BaselineReport, pd.DataFrame]:
    train = answers[answers["split"] == "train"].copy()
    dev = answers[answers["split"] == "dev"].copy()
    if train.empty or dev.empty:
        raise ValueError(f"track {track}: train and dev rows are required")
    selected_c, c_records = select_c(train, dev, mode)
    scores = fit_text_baseline(_expand_multiplicity(train), test, mode, selected_c)
    labels = test["label"].to_numpy(dtype=int)
    overall = float(roc_auc_score(labels, scores))
    within = within_question_auroc(scores, labels, test["question_id"].to_numpy())
    saved = {
        name: float(roc_auc_score(labels, test[column].to_numpy(dtype=float)))
        for name, column in saved_columns.items()
        if column in test.columns
    }
    observed, interval, bootstrap_mean = _subject_bootstrap_delta(
        test, scores, test[hidden_column].to_numpy(dtype=float)
    )
    within_observed, within_interval = _question_bootstrap_delta(
        test, scores, test[hidden_column].to_numpy(dtype=float)
    )
    report = BaselineReport(
        track=track,
        mode=mode,
        n_train_questions=int(train["question_id"].nunique()),
        n_train_rows=int(len(train)),
        n_train_subjects=int(train["subject_id"].nunique()),
        n_dev_questions=int(dev["question_id"].nunique()),
        n_dev_rows=int(len(dev)),
        n_test_questions=int(test["question_id"].nunique()),
        n_test_rows=int(len(test)),
        n_test_subjects=int(test["subject_id"].nunique()),
        selected_c=float(selected_c),
        overall_auroc=overall,
        within_question_auroc=within["within_question_auroc"],
        eligible_questions=within["eligible_questions"],
        saved=saved,
        observed_delta_vs_hidden=observed,
        delta_ci=interval,
        bootstrap_mean_delta=bootstrap_mean,
        within_observed_delta=within_observed,
        within_delta_ci=within_interval,
        provenance={**provenance, "c_selection": {str(k): v for k, v in c_records.items()}},
    )
    score_frame = test[
        ["question_id", "subject_id", "label", "relation", "popularity", "stratum"]
        + [column for column in saved_columns.values() if column in test.columns]
    ].copy()
    score_frame["track"] = track
    score_frame["mode"] = mode
    score_frame["text_score"] = scores
    return report, score_frame


def prepare(run_dir: Path, track: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cohort = load_questions(run_dir)
    answers = load_answers(run_dir, track, cohort)
    question_map = cohort[["question_id", "question"]].drop_duplicates("question_id")
    answers = answers.merge(question_map, on="question_id", how="left")
    predictions = pd.read_parquet(
        run_dir / "predictions" / f"test_predictions__qwen__qwen2.5_7b_instruct__track{track}.parquet"
    )
    predictions = predictions.merge(question_map, on="question_id", how="left")
    if "label" not in predictions.columns:
        raise ValueError("test predictions have no label column")
    files = canonical_answer_files(run_dir, track)
    provenance = {
        "run_dir": str(run_dir),
        "answer_files": [{"path": str(path), "sha256": _sha256(path)} for path in files],
        "cohort_manifest": {
            "path": str(run_dir / "manifests" / "cohort.parquet"),
            "sha256": _sha256(run_dir / "manifests" / "cohort.parquet"),
        },
        "predictions": str(
            run_dir / "predictions" / f"test_predictions__qwen__qwen2.5_7b_instruct__track{track}.parquet"
        ),
        "expected_draws": 1 if track.upper() == "A" else 8,
    }
    return answers, predictions, provenance


def run_track(run_dir: Path, track: str, modes: tuple[str, ...]):
    answers, predictions, provenance = prepare(run_dir, track)
    suffix = "_after" if track.upper() == "B" else ""
    saved_columns = {
        "saved_H": f"cal_H_512{suffix}",
        "saved_P": f"cal_P{suffix}",
        "saved_V": f"cal_V{suffix}",
    }
    hidden_column = saved_columns["saved_H"]
    if hidden_column not in predictions.columns:
        raise ValueError(f"predictions lack {hidden_column}")
    reports, score_frames = [], []
    for mode in modes:
        report, scores = evaluate_baseline(
            track=track,
            mode=mode,
            answers=answers,
            test=predictions,
            saved_columns=saved_columns,
            hidden_column=hidden_column,
            provenance=provenance,
        )
        reports.append(report)
        score_frames.append(scores)
    return reports, pd.concat(score_frames, ignore_index=True)


def save_reports(run_dir: Path, reports: list[BaselineReport], scores: pd.DataFrame) -> Path:
    out_dir = run_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "text_baselines_repaired.json"
    payload = {
        "note": "S0 repaired baselines: frozen cohort, canonical collection, draw counts validated",
        "reports": [report.as_dict() for report in reports],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    scores.to_parquet(out_dir / "text_baselines_repaired_scores.parquet", index=False)
    return path
