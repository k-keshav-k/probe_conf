"""Configuration for the selective-metacognition toy experiments.

The proposal is ``report/SELECTIVE_MLP_METACOGNITION_PROPOSAL.md``.  These
defaults are the development settings from its section 8.1 and are not
optimized values.
"""

from __future__ import annotations

from dataclasses import dataclass

CONFIDENCE_GRID: tuple[float, ...] = tuple(index / 20 for index in range(21))

# Conditions from proposal section 8.2.  ``valid_swap`` changes the declared
# world, so the correct answer changes with it; ``false_memory`` changes the
# memory but keeps the original world.
CONDITIONS: tuple[str, ...] = (
    "clean",
    "irrelevant_swap",
    "target_weaken",
    "irrelevant_weaken",
    "target_swap_valid",
    "false_memory",
    "sham",
)

# Conditions used to train the reporter, so that corruption strength alone
# cannot determine correctness (proposal section 8.3).
REPORT_CONDITIONS: tuple[str, ...] = (
    "clean",
    "target_weaken",
    "irrelevant_weaken",
    "irrelevant_swap",
    "target_swap_valid",
    "false_memory",
)

# Observable-reliability conditions for the primary reporting endpoint.  The
# coherent false memory is evaluated separately: its correct answer is not
# identifiable from the input, so it cannot be part of the main task.
OBSERVABLE_CONDITIONS: tuple[str, ...] = (
    "clean",
    "target_weaken",
    "irrelevant_weaken",
    "irrelevant_swap",
    "target_swap_valid",
)
FALSE_MEMORY_CONDITIONS: tuple[str, ...] = ("false_memory",)


@dataclass
class ToyConfig:
    # --- world -------------------------------------------------------------
    n_subjects: int = 512
    n_relations: int = 4
    n_answers: int = 128
    code_dim: int = 64
    memory_dim: int = 256
    train_subjects: int = 384
    dev_subjects: int = 64
    test_subjects: int = 64
    weaken_strength: float = 0.25
    overlap: float = 0.0
    dense: bool = False
    spurious: float = 0.0
    # --- model -------------------------------------------------------------
    d_model: int = 128
    n_heads: int = 4
    d_hidden: int = 256
    n_layers: int = 2
    adapter_rank: int = 8
    adapter_layers: tuple[int, ...] = ()  # empty = last block
    d_report: int = 256
    report_layers: int = 2
    temperature: float = 1.0
    # --- retrieval training ------------------------------------------------
    batch: int = 128
    warm_start_steps: int = 1500
    warm_start_lr: float = 3e-3
    steps: int = 800
    lr: float = 1e-3
    beta_kl: float = 0.1
    lambda_sel: float = 1.0
    eta_consistency: float = 1.0
    # --- reporting ---------------------------------------------------------
    confidence_grid: tuple[float, ...] = CONFIDENCE_GRID
    report_steps: int = 3000
    report_lr: float = 1e-3
    report_batch: int = 256
    report_collect_steps: int = 120
    outcomes_per_item: int = 8
    eval_outcomes: int = 16
    rounds: int = 3
    # --- misc --------------------------------------------------------------
    seeds: int = 3

    def __post_init__(self) -> None:
        if not self.dense and self.n_relations * self.code_dim > self.memory_dim:
            raise ValueError("block mode needs n_relations * code_dim <= memory_dim")
        if self.train_subjects + self.dev_subjects + self.test_subjects > self.n_subjects:
            raise ValueError("train + dev + test subjects exceed n_subjects")

    def as_dict(self) -> dict:
        return dict(self.__dict__)
