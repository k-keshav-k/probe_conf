"""Probe fitting with fold-internal preprocessing.

Plan requirements implemented here:

- scalers, imputation, PCA, and the classifier fit inside every training fold;
- subject-grouped folds (outer evaluation and inner model selection);
- three grouped inner folds, selection by mean Brier score with AUROC as the
  fixed tie-breaker;
- a fixed C grid and a hard configuration budget (24 per side);
- a matched 512-dimensional internal-feature budget and the additive controls;
- a small nonlinear readout with a frozen schedule;
- fixed probe seeds 0-4 for stochastic readouts (predictions averaged).

Feature columns are named ``f__{family}__{stat}__{position}__l{layer}``.  A
``Block`` selects a subset and gives it a dimension budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import PROTOCOL
from ..utils import set_seed


# ---------------------------------------------------------------------------
# Feature blocks
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Block:
    name: str
    include: tuple[str, ...]  # column-name prefixes that belong to the block
    positions: tuple[str, ...] = ("last_prompt",)
    layers: tuple[int, ...] | None = None
    pca_dim: int | None = None

    def matches(self, column: str) -> bool:
        if not column.startswith("f__"):
            return False
        body = column[3:]
        if not any(body.startswith(prefix) for prefix in self.include):
            return False
        if self.positions and not any(f"__{position}__" in body for position in self.positions):
            return False
        if self.layers is not None and not any(body.endswith(f"__l{layer}") for layer in self.layers):
            return False
        return True


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    blocks: tuple[Block, ...]
    classifier: str = "logreg"  # "logreg" or "mlp"


def select_columns(df, block: Block) -> list[str]:
    return [column for column in df.columns if block.matches(column)]


def assemble_blocks(df, spec: ProbeSpec) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for block in spec.blocks:
        columns = select_columns(df, block)
        if not columns:
            matrices[block.name] = np.zeros((len(df), 0), dtype=np.float32)
            continue
        matrix = df[columns].to_numpy(dtype=np.float32, copy=True)
        matrices[block.name] = matrix
    return matrices


# ---------------------------------------------------------------------------
# Per-block preprocessing, fitted on the training fold only
# ---------------------------------------------------------------------------
class BlockPreprocessor:
    def __init__(self, pca_dim: int | None = None):
        self.pca_dim = pca_dim
        self.imputer: SimpleImputer | None = None
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.columns: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "BlockPreprocessor":
        x = np.asarray(x, dtype=np.float32)
        self.columns = np.arange(x.shape[1])
        if x.shape[1] == 0:
            return self
        nan_columns = np.isnan(x).any(axis=0)
        if nan_columns.any():
            indicator = np.isnan(x[:, nan_columns]).astype(np.float32)
            self.columns = np.concatenate([self.columns, np.where(nan_columns)[0]])
            x = np.concatenate([x, indicator], axis=1)
        self.imputer = SimpleImputer(strategy="median").fit(x)
        x = self.imputer.transform(x)
        self.scaler = StandardScaler().fit(x)
        x = self.scaler.transform(x).astype(np.float32)
        if self.pca_dim is not None and x.shape[1] > self.pca_dim:
            # PCA's full solver requires n_components <= min(n_samples, n_features).
            # Small development folds can be shorter than the component budget.
            n_components = min(self.pca_dim, x.shape[0])
            self.pca = PCA(n_components=n_components, random_state=0).fit(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.scaler is None:  # empty block
            return np.zeros((len(x), 0), dtype=np.float32)
        if x.shape[1] > 0 and np.isnan(x).any():
            nan_columns = np.isnan(x).any(axis=0)
            indicator = np.isnan(x[:, nan_columns]).astype(np.float32)
            x = np.concatenate([x, indicator], axis=1)
        x = self.imputer.transform(x)
        x = self.scaler.transform(x).astype(np.float32)
        if self.pca is not None:
            x = self.pca.transform(x)
        return x


class FittedProbe:
    """A fitted probe: preprocessing per block plus one classifier."""

    def __init__(self, spec: ProbeSpec, preprocessors: dict, classifier, seed: int):
        self.spec = spec
        self.preprocessors = preprocessors
        self.classifier = classifier
        self.seed = seed

    def _features(self, matrices: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for block in self.spec.blocks:
            parts.append(self.preprocessors[block.name].transform(matrices[block.name]))
        parts = [p for p in parts if p.shape[1] > 0]
        if not parts:
            return np.zeros((len(next(iter(matrices.values()))), 0), dtype=np.float32)
        return np.concatenate(parts, axis=1)

    def predict(self, matrices: dict[str, np.ndarray]) -> np.ndarray:
        x = self._features(matrices)
        if x.shape[1] == 0:
            return np.full(len(x), 0.5, dtype=np.float32)
        return self.classifier.predict_proba(x)[:, 1]


# ---------------------------------------------------------------------------
# Configuration search
# ---------------------------------------------------------------------------
def default_configs(classifier: str, budget: int = PROTOCOL.max_configs_per_side) -> list[dict]:
    if classifier == "logreg":
        configs = [{"C": c, "max_iter": 3000} for c in PROTOCOL.c_grid]
    elif classifier == "mlp":
        configs = [
            {
                "hidden_layer_sizes": (PROTOCOL.nonlinear_hidden,),
                "learning_rate_init": lr,
                "weight_decay": wd,
                "max_iter": PROTOCOL.nonlinear_epochs,
            }
            for lr in PROTOCOL.nonlinear_lr
            for wd in PROTOCOL.nonlinear_wd
        ]
    else:
        raise ValueError(classifier)
    return configs[:budget]


def _make_classifier(classifier: str, config: dict, seed: int):
    if classifier == "logreg":
        return LogisticRegression(C=config["C"], max_iter=config["max_iter"])
    return MLPClassifier(
        hidden_layer_sizes=config["hidden_layer_sizes"],
        learning_rate_init=config["learning_rate_init"],
        alpha=config["weight_decay"],
        max_iter=config["max_iter"],
        early_stopping=True,
        n_iter_no_change=PROTOCOL.nonlinear_patience,
        random_state=seed,
    )


def _fit_once(matrices: dict, y: np.ndarray, spec: ProbeSpec, config: dict, seed: int) -> FittedProbe:
    preprocessors = {}
    for block in spec.blocks:
        pre = BlockPreprocessor(pca_dim=block.pca_dim).fit(matrices[block.name])
        preprocessors[block.name] = pre
    probe = FittedProbe(spec, preprocessors, None, seed)
    x = probe._features(matrices)
    if x.shape[1] == 0:
        # No matching features in this frame (for example a last-prompt spec on
        # an answer-position-only Track B frame).  Predict a constant 0.5.
        return probe
    classifier = _make_classifier(spec.classifier, config, seed)
    classifier.fit(x, y)
    probe.classifier = classifier
    return probe


def fit_probe(
    df,
    y,
    groups,
    spec: ProbeSpec,
    configs: list[dict] | None = None,
    seed: int = 0,
) -> tuple[FittedProbe, dict]:
    """Select a configuration with grouped inner CV, then fit on all training rows."""

    matrices = assemble_blocks(df, spec)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    configs = configs or default_configs(spec.classifier)
    set_seed(seed)

    inner = StratifiedGroupKFold(n_splits=PROTOCOL.inner_folds, shuffle=True, random_state=seed)
    best = None
    for config in configs:
        scores = []
        for train_index, valid_index in inner.split(np.zeros(len(y)), y, groups):
            sub = {name: matrix[train_index] for name, matrix in matrices.items()}
            probe = _fit_once(sub, y[train_index], spec, config, seed)
            valid = {name: matrix[valid_index] for name, matrix in matrices.items()}
            prediction = probe.predict(valid)
            scores.append(
                (
                    brier_score_loss(y[valid_index], np.clip(prediction, 0, 1)),
                    -roc_auc_score(y[valid_index], prediction),
                )
            )
        mean_brier = float(np.mean([s[0] for s in scores]))
        mean_auc = float(-np.mean([s[1] for s in scores]))
        record = {"config": config, "brier": mean_brier, "auroc": mean_auc}
        if best is None or (mean_brier, -mean_auc) < (best["brier"], -best["auroc"]):
            best = record
    probe = _fit_once(matrices, y, spec, best["config"], seed)
    return probe, best


def seeded_predict(
    df,
    y,
    groups,
    spec: ProbeSpec,
    seeds: tuple[int, ...] = PROTOCOL.seeds,
) -> tuple[np.ndarray, list[dict]]:
    """Fit the probe with fixed seeds and average predictions.

    Logistic regression is deterministic; the loop still returns one prediction
    per seed so the sensitivity analysis is uniform.
    """

    matrices = assemble_blocks(df, spec)
    predictions = []
    records = []
    for seed in seeds:
        probe, best = fit_probe(df, y, groups, spec, seed=seed)
        predictions.append(probe.predict(matrices))
        records.append({"seed": seed, **best})
    return np.mean(predictions, axis=0), records


# ---------------------------------------------------------------------------
# Frozen comparison feature sets (plan section 6.1)
# ---------------------------------------------------------------------------
def standard_specs() -> dict[str, ProbeSpec]:
    last = ("last_prompt",)
    after = ("answer_last_content",)
    meta_block = Block("meta", ("meta__",), positions=())
    prob_block = Block("prob", ("prob__",), positions=last)
    prob_block_after = Block("prob_after", ("prob__",), positions=after)
    hidden_block = Block("hproj", ("hiddenproj__post__", "hiddenproj__pre__"), positions=last)
    hidden_block_after = Block("hproj_after", ("hiddenproj__post__", "hiddenproj__pre__"), positions=after)
    mlp_all = (
        Block("m_gates", ("gate_selection__",), positions=last),
        Block("m_conc", ("contribution_concentration__",), positions=last),
        Block("m_dir", ("output_direction__",), positions=last),
        Block("m_lit", ("w2_literal__",), positions=last),
    )
    mlp_all_after = (
        Block("m_gates", ("gate_selection__",), positions=after),
        Block("m_conc", ("contribution_concentration__",), positions=after),
        Block("m_dir", ("output_direction__",), positions=after),
        Block("m_lit", ("w2_literal__",), positions=after),
    )
    return {
        "M_concentration": ProbeSpec(
            "M_concentration",
            (Block("m_conc", ("contribution_concentration__",), positions=last),),
        ),
        "M_gates": ProbeSpec(
            "M_gates",
            (Block("m_gates", ("gate_selection__",), positions=last),),
        ),
        "M_literal": ProbeSpec(
            "M_literal",
            (Block("m_lit", ("w2_literal__",), positions=last),),
        ),
        "M_direction": ProbeSpec(
            "M_direction",
            (Block("m_dir", ("output_direction__",), positions=last),),
        ),
        "M_all": ProbeSpec("M_all", mlp_all),
        "M_all_after": ProbeSpec("M_all_after", mlp_all_after),
        "P": ProbeSpec("P", (prob_block,)),
        "P_after": ProbeSpec("P_after", (prob_block_after,)),
        "V": ProbeSpec("V", (Block("v", ("conf__",), positions=()),)),
        "V_after": ProbeSpec("V_after", (Block("v", ("conf__",), positions=()),)),
        "H_proj": ProbeSpec(
            "H_proj",
            (Block("h", ("hiddenproj__post__",), positions=last, pca_dim=PROTOCOL.pca_budget),),
        ),
        "H_512": ProbeSpec(
            "H_512",
            (Block("h", ("hiddenproj__post__", "hiddenproj__pre__"), positions=last, pca_dim=PROTOCOL.pca_budget),),
        ),
        "H_512_after": ProbeSpec(
            "H_512_after",
            (Block("h", ("hiddenproj__post__", "hiddenproj__pre__"), positions=after, pca_dim=PROTOCOL.pca_budget),),
        ),
        "H_M_256_256": ProbeSpec(
            "H_M_256_256",
            (
                Block("h", ("hiddenproj__post__", "hiddenproj__pre__"), positions=last, pca_dim=PROTOCOL.pca_hidden_half),
                Block("m_conc", ("contribution_concentration__",), positions=last, pca_dim=PROTOCOL.pca_mlp_half),
            ),
        ),
        "H_all_plus_M": ProbeSpec(
            "H_all_plus_M",
            (Block("h", ("hiddenproj__post__", "hiddenproj__pre__"), positions=last),) + mlp_all,
        ),
        "H_all_plus_M_after": ProbeSpec(
            "H_all_plus_M_after",
            (Block("h", ("hiddenproj__post__", "hiddenproj__pre__"), positions=after),) + mlp_all_after,
        ),
        "H_768": ProbeSpec(
            "H_768",
            (
                Block(
                    "h",
                    ("hiddenproj__post__", "hiddenproj__pre__"),
                    positions=last,
                    pca_dim=PROTOCOL.pca_budget + PROTOCOL.pca_mlp_half,
                ),
            ),
        ),
        "H_nonlinear": ProbeSpec(
            "H_nonlinear",
            (
                Block(
                    "h",
                    ("hiddenproj__post__", "hiddenproj__pre__"),
                    positions=last,
                    pca_dim=PROTOCOL.nonlinear_hidden,
                ),
            ),
            classifier="mlp",
        ),
        "B": ProbeSpec("B", (meta_block, prob_block, hidden_block)),
        "B_plus_M": ProbeSpec("B_plus_M", (meta_block, prob_block, hidden_block) + mlp_all),
        "B_after": ProbeSpec("B_after", (meta_block, prob_block_after, hidden_block_after)),
        "B_plus_M_after": ProbeSpec(
            "B_plus_M_after", (meta_block, prob_block_after, hidden_block_after) + mlp_all_after
        ),
    }
