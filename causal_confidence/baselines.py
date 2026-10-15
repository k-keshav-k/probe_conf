"""Question/probability/hidden-state predictors, fitted out of sample.

Every predictor declares its information access (plan section 11.1):

- constant (training success rate; also constant 0.5),
- question-only metadata (relation, popularity, length),
- prefix probability summaries at the answer fork,
- rich prefix (top-k logits plus hashed token identities),
- hidden-state readout of the retained prefix state,
- generated-answer comparator (answer statistics),
- sampling comparator (agreement across draws; a reference estimate),
- privileged diagnostic (condition identity and strength; never presented as
  deployable).

Ordinary baselines never receive condition identity.  Selection of the
strongest ordinary baseline happens on development data, never on test.
Fitting and the baseline-plus-report refits use the same model classes and
budgets so added value is not a capacity artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .protocol import PROTOCOL


@dataclass
class BaselineFeatures:
    """Feature blocks for one set of rows.  Blocks are aligned by row."""

    blocks: dict[str, np.ndarray] = field(default_factory=dict)

    def with_blocks(self, **kwargs) -> "BaselineFeatures":
        merged = dict(self.blocks)
        merged.update(kwargs)
        return BaselineFeatures(blocks=merged)

    def stack(self, names: list[str]) -> np.ndarray:
        parts = [self.blocks[name] for name in names if name in self.blocks]
        if not parts:
            raise KeyError(f"none of {names} are present in {sorted(self.blocks)}")
        return np.concatenate(parts, axis=1).astype(np.float32)


ORDINARY_BLOCKS = {
    "constant_rate": [],
    "constant_half": [],
    "question": ["question"],
    "prefix": ["question", "prefix"],
    "rich": ["question", "rich"],
    "hidden": ["question", "hidden"],
    "answer": ["question", "answer"],
    "sampling": ["sampling"],
}


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------
def question_features(
    relations: list[str],
    popularity: list[float | None],
    question_texts: list[str],
    *,
    relation_order: tuple[str, ...] = PROTOCOL.relations,
) -> np.ndarray:
    """Relation one-hot, log popularity (with a missing indicator), lengths."""

    relation_index = {name: index for index, name in enumerate(relation_order)}
    rows = []
    for relation, pop, text in zip(relations, popularity, question_texts):
        onehot = [0.0] * len(relation_order)
        if relation in relation_index:
            onehot[relation_index[relation]] = 1.0
        missing = 0.0 if pop is None or (isinstance(pop, float) and np.isnan(pop)) else 1.0
        value = float(np.log1p(pop)) if missing else 0.0
        length = min(len(text.split()), 64) / 64.0
        rows.append(onehot + [missing, value, length, len(text) / 512.0])
    return np.asarray(rows, dtype=np.float32)


def prefix_probability_features(logits: np.ndarray) -> np.ndarray:
    """Top-1 probability, entropy, top-2 margin, top-10/top-50 mass."""

    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    ordered = np.sort(probabilities, axis=-1)[:, ::-1]
    top1 = ordered[:, 0]
    top2 = ordered[:, 1] if ordered.shape[1] > 1 else np.zeros_like(top1)
    margin = top1 - top2
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum(axis=-1)
    top10 = ordered[:, :10].sum(axis=-1) if ordered.shape[1] >= 10 else np.ones_like(top1)
    top50 = ordered[:, :50].sum(axis=-1) if ordered.shape[1] >= 50 else np.ones_like(top1)
    return np.stack(
        [top1, top2, margin, entropy / max(1.0, np.log(logits.shape[-1])), top10, top50],
        axis=1,
    ).astype(np.float32)


def rich_prefix_features(logits: np.ndarray, *, k: int = 32, hash_dim: int = 64) -> np.ndarray:
    """Top-k logits plus hashed token identities carrying the logit value."""

    logits = np.asarray(logits, dtype=np.float32)
    k = min(k, logits.shape[-1])
    order = np.argsort(-logits, axis=-1)[:, :k]
    values = np.take_along_axis(logits, order, axis=-1)
    hashed = np.zeros((logits.shape[0], hash_dim), dtype=np.float32)
    for row in range(logits.shape[0]):
        for rank in range(k):
            token = int(order[row, rank])
            slot = token % hash_dim
            hashed[row, slot] += float(values[row, rank])
    return np.concatenate([values, hashed], axis=1).astype(np.float32)


def hidden_features(hidden: np.ndarray, *, projection_dim: int = 256, seed: int = 20261015) -> np.ndarray:
    """Seeded fixed projection of a retained hidden state.

    A fixed projection keeps the baseline storage bounded and avoids fitting
    the projection on outcome labels.
    """

    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim == 1:
        hidden = hidden[None, :]
    if hidden.shape[1] <= projection_dim:
        return hidden.astype(np.float32)
    generator = np.random.default_rng(seed)
    projection = generator.standard_normal((hidden.shape[1], projection_dim)).astype(
        np.float32
    ) / np.sqrt(hidden.shape[1])
    return (hidden @ projection).astype(np.float32)


def answer_features(
    texts: list[str], logprobs: list[float], n_tokens: list[int]
) -> np.ndarray:
    """Statistics of a separately generated answer."""

    rows = []
    for text, logprob, tokens in zip(texts, logprobs, n_tokens):
        chars = len(text)
        words = len(text.split())
        has_digit = float(any(character.isdigit() for character in text))
        rows.append(
            [
                min(tokens, 64) / 64.0,
                min(logprob / max(tokens, 1), 0.0) if tokens else 0.0,
                min(chars, 128) / 128.0,
                min(words, 16) / 16.0,
                has_digit,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def sampling_features(bank_success: list[float]) -> np.ndarray:
    """Agreement across draws: the empirical success of sampled answers."""

    return np.asarray(bank_success, dtype=np.float32)[:, None]


def privileged_features(slots: list[str], strengths: list[float]) -> np.ndarray:
    """Condition identity and strength.  Diagnostic only, never deployable."""

    from .protocol import CONDITION_SLOTS

    index = {name: position for position, name in enumerate(CONDITION_SLOTS)}
    rows = []
    for slot, strength in zip(slots, strengths):
        onehot = [0.0] * len(CONDITION_SLOTS)
        onehot[index.get(slot, 0)] = 1.0
        rows.append(onehot + [float(strength)])
    return np.asarray(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _make_pipeline(kind: str, seed: int):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logistic":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1_000, C=1.0, random_state=seed),
                ),
            ]
        )
    if kind == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    raise ValueError(f"unknown model kind: {kind!r}")


def fit_predict_outcomes(
    x_train: np.ndarray,
    outcomes_train: np.ndarray,
    x_test: np.ndarray,
    *,
    kind: str = "logistic",
    seed: int = 0,
) -> np.ndarray:
    """Fit on outcome draws (expanded for logistic) and predict probabilities.

    ``outcomes_train`` has shape ``(n, K)``.  Logistic models expand rows so
    the loss matches the Bernoulli draws; ridge models fit the draw mean.
    """

    x_train = np.asarray(x_train, dtype=np.float32)
    outcomes_train = np.asarray(outcomes_train, dtype=np.float32)
    if kind == "logistic":
        expanded = np.repeat(x_train, outcomes_train.shape[1], axis=0)
        labels = outcomes_train.reshape(-1)
        model = _make_pipeline(kind, seed)
        model.fit(expanded, labels)
        return model.predict_proba(np.asarray(x_test, dtype=np.float32))[:, 1]
    model = _make_pipeline(kind, seed)
    model.fit(x_train, outcomes_train.mean(axis=1))
    return model.predict(np.asarray(x_test, dtype=np.float32))


def baseline_predictions(
    train: BaselineFeatures,
    outcomes_train: np.ndarray,
    test: BaselineFeatures,
    *,
    seed: int = 0,
    include: tuple[str, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Predictions for every available ordinary baseline plus constants."""

    outcomes_train = np.asarray(outcomes_train, dtype=np.float32)
    names = include or tuple(ORDINARY_BLOCKS)
    predictions: dict[str, np.ndarray] = {}
    if test.blocks:
        n_test = len(next(iter(test.blocks.values())))
    else:
        raise ValueError("test features must contain at least one block")
    for name in names:
        if name == "constant_rate":
            predictions[name] = np.full(
                (n_test,), float(outcomes_train.mean()), dtype=np.float32
            )
        elif name == "constant_half":
            predictions[name] = np.full((n_test,), 0.5, dtype=np.float32)
        elif name == "sampling":
            if "sampling" not in test.blocks:
                continue
            predictions[name] = test.blocks["sampling"][:, 0].astype(np.float32)
        else:
            blocks = ORDINARY_BLOCKS[name]
            if any(block not in train.blocks or block not in test.blocks for block in blocks):
                continue
            kind = "ridge" if name == "hidden" else "logistic"
            predictions[name] = fit_predict_outcomes(
                train.stack(blocks),
                outcomes_train,
                test.stack(blocks),
                kind=kind,
                seed=seed,
            ).astype(np.float32)
    if not predictions:
        raise ValueError("no baseline features were provided")
    return predictions


def fit_predict_augmented(
    train: BaselineFeatures,
    outcomes_train: np.ndarray,
    test: BaselineFeatures,
    report_train: np.ndarray,
    report_test: np.ndarray,
    *,
    baseline: str,
    seed: int = 0,
) -> np.ndarray:
    """Refit one baseline with the report as an additional input.

    The model class and budget match the plain baseline; the only change is
    the extra column.  This is the fitted ``b + c`` comparison, not numerical
    addition of two predictions.
    """

    blocks = ORDINARY_BLOCKS[baseline]
    x_train = np.concatenate(
        [train.stack(blocks), np.asarray(report_train, dtype=np.float32)[:, None]], axis=1
    )
    x_test = np.concatenate(
        [test.stack(blocks), np.asarray(report_test, dtype=np.float32)[:, None]], axis=1
    )
    kind = "ridge" if baseline == "hidden" else "logistic"
    return fit_predict_outcomes(
        x_train, outcomes_train, x_test, kind=kind, seed=seed
    ).astype(np.float32)


def select_strongest_baseline(
    predictions: dict[str, np.ndarray], outcomes: np.ndarray
) -> tuple[str, dict[str, float]]:
    """Pick the ordinary baseline with the best development Brier."""

    from .evaluate import brier

    scores = {
        name: brier(values, outcomes) for name, values in predictions.items()
    }
    strongest = min(scores, key=scores.get)
    return strongest, scores
