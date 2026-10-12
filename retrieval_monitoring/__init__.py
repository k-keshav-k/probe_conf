"""Retrieval monitoring: can a model tell when its factual recall is unreliable?

This package implements the experiment plan in
``report/RETRIEVAL_MONITORING_PLAN.md``.  It is separate from the Section 3.3
confirmatory code in ``section33/``.

Modules
-------
- ``retained_state``: run a prefix once, fork the KV cache into an answer branch
  and a confidence branch, with correct attention-mask handling.
- ``text_baselines``: repaired S0 input-only baselines on the frozen cohort.
- ``interventions``: relation maps and matched random controls.
- ``evaluation``: the paired Brier endpoint and its controls.
- ``toy_world``: S2 one-layer toy with embedding-level interventions.
- ``collector``: retained-state generation and scoring.
- ``pilot``: S3 40-subject selectivity pilot.

Entry points live in ``retrieval_monitoring.cli``:

    python -m retrieval_monitoring.cli text-baselines --run runs/section33_confirmatory
    python -m retrieval_monitoring.cli check-state --model Qwen/Qwen2.5-0.5B-Instruct
    python -m retrieval_monitoring.cli toy
    python -m retrieval_monitoring.cli pilot --run runs/section33_confirmatory
"""

__version__ = "0.1.0"
