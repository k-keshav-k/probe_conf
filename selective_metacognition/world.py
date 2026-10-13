"""Controlled factual world with known relation components.

Each subject is a sum of relation components::

    h_s = sum_r T_r v_{a(s, r)}

``v_a`` is a unit answer code and ``T_r`` embeds it into a disjoint block of
the memory, so the selector can in principle read the queried relation's
component without interference.  Conditions change known components, which
gives exact ground truth for every intervention (proposal section 8).

The evaluator knows the construction.  The model never sees condition labels,
answer identifiers, or the relation of an unqueried change.
"""

from __future__ import annotations

import torch

from .config import ToyConfig


class World:
    def __init__(
        self,
        config: ToyConfig,
        seed: int,
        assignments: torch.Tensor | None = None,
        device: torch.device | str | None = None,
    ):
        self.config = config
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        generator = torch.Generator().manual_seed(seed)
        codes = torch.randn(config.n_answers, config.code_dim, generator=generator)
        self.answer_codes = (codes / codes.norm(dim=1, keepdim=True)).to(self.device)

        maps = torch.zeros(config.n_relations, config.memory_dim, config.code_dim)
        for relation in range(config.n_relations):
            block = torch.randn(config.code_dim, config.code_dim, generator=generator)
            q, _ = torch.linalg.qr(block)
            start = relation * config.code_dim
            maps[relation, start : start + config.code_dim, :] = q
        self.relation_maps = maps.to(self.device)

        self.assignments = (
            assignments
            if assignments is not None
            else torch.randint(
                0, config.n_answers, (config.n_subjects, config.n_relations), generator=generator
            )
        ).to(self.device)
        self.subject_ids = torch.arange(config.n_subjects, device=self.device)

    # -- splits -------------------------------------------------------------
    @property
    def train_ids(self) -> torch.Tensor:
        return torch.arange(self.config.train_subjects)

    @property
    def dev_ids(self) -> torch.Tensor:
        start = self.config.train_subjects
        return torch.arange(start, start + self.config.dev_subjects)

    @property
    def test_ids(self) -> torch.Tensor:
        start = self.config.train_subjects + self.config.dev_subjects
        return torch.arange(start, start + self.config.test_subjects)

    def resample_assignments(self, seed: int) -> torch.Tensor:
        """Fresh bindings for the same subjects and the same answer codes."""

        generator = torch.Generator().manual_seed(seed)
        return torch.randint(
            0, self.config.n_answers, (self.config.n_subjects, self.config.n_relations), generator=generator
        ).to(self.device)

    # -- memory -------------------------------------------------------------
    def component(self, answer_ids: torch.Tensor, relation: int) -> torch.Tensor:
        """``T_r v_a`` for a batch of answer ids."""

        return torch.einsum("dc,nc->nd", self.relation_maps[relation], self.answer_codes[answer_ids])

    def clean_memory(
        self, subject_ids: torch.Tensor, assignments: torch.Tensor | None = None
    ) -> torch.Tensor:
        assignments = self.assignments if assignments is None else assignments
        subject_ids = subject_ids.to(assignments.device)
        answers = assignments[subject_ids]
        memory = torch.zeros(len(subject_ids), self.config.memory_dim, device=assignments.device)
        for relation in range(self.config.n_relations):
            memory = memory + self.component(answers[:, relation], relation)
        return memory

    # -- conditions ---------------------------------------------------------
    @staticmethod
    def _other_relation(queried: int, generator: torch.Generator, n_relations: int) -> int:
        choices = [relation for relation in range(n_relations) if relation != queried]
        index = int(torch.randint(len(choices), (1,), generator=generator))
        return choices[index]

    @staticmethod
    def _alternate_answers(current: torch.Tensor, generator: torch.Generator, n_answers: int) -> torch.Tensor:
        alternate = torch.randint(0, n_answers, current.shape, generator=generator).to(current.device)
        return torch.where(alternate == current, (alternate + 1) % n_answers, alternate)

    def condition(
        self,
        subject_ids: torch.Tensor,
        queried_relation: int | torch.Tensor,
        name: str,
        *,
        strength: float | None = None,
        seed: int = 0,
        assignments: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(memory, correct_answer)`` under a named condition.

        ``queried_relation`` may be an integer or one relation per subject.
        ``correct_answer`` is the answer the declared world expects: for
        ``target_swap_valid`` it is the new answer, otherwise the original one.
        """

        assignments = self.assignments if assignments is None else assignments
        strength = self.config.weaken_strength if strength is None else strength
        subject_ids = subject_ids.to(assignments.device)
        if isinstance(queried_relation, int):
            queried_relation = torch.full(
                (len(subject_ids),), queried_relation, dtype=torch.long, device=self.device
            )
        queried_relation = queried_relation.to(self.device)
        answers = assignments[subject_ids]
        memory = self.clean_memory(subject_ids, assignments)
        correct = answers[torch.arange(len(subject_ids), device=self.device), queried_relation].clone()
        if name == "clean":
            return memory, correct

        generator = torch.Generator().manual_seed(seed)
        other = torch.tensor(
            [
                self._other_relation(int(relation), generator, self.config.n_relations)
                for relation in queried_relation
            ],
            device=self.device,
        )
        row = torch.arange(len(subject_ids), device=self.device)

        if name == "irrelevant_swap":
            current = answers[row, other]
            alternate = self._alternate_answers(current, generator, self.config.n_answers)
            memory = memory - self._component_per_subject(current, other) + self._component_per_subject(alternate, other)
        elif name == "target_weaken":
            current = answers[row, queried_relation]
            memory = memory - (1.0 - strength) * self._component_per_subject(current, queried_relation)
        elif name == "irrelevant_weaken":
            current = answers[row, other]
            memory = memory - (1.0 - strength) * self._component_per_subject(current, other)
        elif name == "target_swap_valid":
            current = answers[row, queried_relation]
            alternate = self._alternate_answers(current, generator, self.config.n_answers)
            memory = memory - self._component_per_subject(current, queried_relation) + self._component_per_subject(
                alternate, queried_relation
            )
            correct = alternate
        elif name == "false_memory":
            current = answers[row, queried_relation]
            alternate = self._alternate_answers(current, generator, self.config.n_answers)
            memory = memory - self._component_per_subject(current, queried_relation) + self._component_per_subject(
                alternate, queried_relation
            )
        elif name == "sham":
            current = answers[row, other]
            delta = (1.0 - strength) * self._component_per_subject(current, other)
            random = torch.randn(memory.shape, generator=generator).to(memory.device)
            random = random / random.norm(dim=1, keepdim=True) * delta.norm(dim=1, keepdim=True)
            memory = memory - delta + random
        else:
            raise ValueError(f"unknown condition: {name}")
        return memory, correct

    def _component_per_subject(self, answer_ids: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """``T_r v_a`` with one relation per row."""

        output = torch.zeros(len(answer_ids), self.config.memory_dim, device=self.device)
        for relation in range(self.config.n_relations):
            mask = relations == relation
            if bool(mask.any()):
                output[mask] = self.component(answer_ids[mask], relation)
        return output
