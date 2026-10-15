"""Causal confidence training: implementation of ``report/CAUSAL_CONFIDENCE_TRAINING_PLAN.md``.

This package is separate from the completed toy experiments in
``selective_metacognition`` and from the S3 relation-map runner in
``section33``.  It does not modify their semantics.

Stages (plan section 10):

- P0: reporter positive control on saved or regenerated diagnostic rows.
- P1: pretrained episode contract, branch isolation, and answer-policy audit.
- P2: bounded discovery and one locked validation of a retrieval manipulation.
- P3-P6: reporter training, transfer, mechanism, and optimizer studies.  These
  are gated on P2 and only their contracts are implemented here.
"""

from .protocol import PROTOCOL, PROTOCOL_VERSION, run_paths  # noqa: F401
