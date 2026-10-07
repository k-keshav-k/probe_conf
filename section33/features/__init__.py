"""Feature contracts for the Section 3.3 experiment."""

from .families import (  # noqa: F401
    FAMILY_REGISTRY,
    REFERENCE_DEPENDENT_NAMES,
    assert_reference_free,
    build_all,
    build_family,
)
from .mlp_math import MLPTensors, mlp_activations  # noqa: F401
