"""Difficulty control, evaluation, and interventions."""

from .difficulty import (  # noqa: F401
    Strata,
    cross_fitted_binomial,
    cross_fitted_logistic,
    fit_binomial_logistic,
    fit_difficulty_score_strata,
    fit_popularity_strata,
    masked_auroc,
    predict_binomial,
    residual_correlation,
    within_question_auroc,
)
from .evaluation import (  # noqa: F401
    Decision,
    auroc,
    bootstrap_p_value,
    calibration_guardrail_interval,
    clip_probabilities,
    decide_equivalence,
    decide_primary,
    decide_within_question,
    holm,
    log_loss,
    paired_subject_bootstrap,
    paired_within_question_bootstrap,
)
from .interventions import (  # noqa: F401
    InterventionResult,
    attenuate_topk,
    flatten_sharpen,
    identity,
    permuted_labels,
    random_matched,
    shuffle_within_strata,
    top_channels,
)
