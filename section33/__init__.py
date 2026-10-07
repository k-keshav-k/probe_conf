"""Section 3.3 confirmatory experiment — implementation package.

This package implements the design in ``report/SECTION_3_3_TEST_PLAN.md``.
It creates no results by itself.  The run artifacts live under
``runs/section33_v1/`` and are written by the ``section33.cli`` entry points.

Layout of the package:

- ``config``      frozen protocol: margins, splits, prompts, budgets.
- ``utils``       hashing, seeding, JSON/JSONL helpers.
- ``features``    MLP mathematics and frozen feature families.
- ``models``      model loading, hook contracts, collection tracks A and B.
- ``data``        cohort construction, exclusion, quotas, split manifests, labels.
- ``probes``      fold-internal preprocessing, readout search, calibration.
- ``analysis``    difficulty controls, paired bootstrap, decision rules, interventions.
- ``cli``         command line entry points for the cluster run.
- ``smoke``       CPU and small-GPU smoke checks (safe to run locally).
"""

__version__ = "0.1.0"
