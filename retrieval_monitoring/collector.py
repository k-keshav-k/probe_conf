"""Retained-state generation and scoring for the retrieval-monitoring plan.

One prefix ("Subject: <name>.") runs per intervention condition.  The cache is
forked into an answer branch and a prospective confidence branch.  Neither
branch receives the intervention label, a sampled answer, or the gold answer.

The intervention acts on the subject-token state at a middle decoder layer, so
the change reaches the keys and values that both branches read.  The collector
is resumable: completed (subject, question, queried relation, state, draw) keys
are skipped.
"""

from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import torch

from section33.data.labels import match_candidate
from section33.models.collect import parse_confidence

from .interventions import (
    RelationMap,
    attenuation_delta,
    matched_random_delta,
)
from .retained_state import (
    additive_transform,
    generate_branch,
    locate_span,
    run_prefix,
)

STATES = ("clean", "relevant", "other", "random")


@dataclass
class CollectorSettings:
    layer_index: int
    alpha: float = 0.5
    n_draws: int = 8
    max_new_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    confidence_tokens: int = 8
    seed: int = 0
    sample_answers: bool = True
    batch_subjects: int = 1

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def subject_prefix(name: str) -> str:
    return f"Subject: {name}."


def answer_suffix(question: str) -> str:
    return f"\nQuestion: {question}\nAnswer:"


def confidence_suffix(question: str) -> str:
    return (
        f"\nQuestion: {question}\n"
        "Confidence that the specified answer policy will be correct (0-100):"
    )


def _encode(tokenizer, text: str, device) -> torch.Tensor:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return torch.tensor([ids], dtype=torch.long, device=device)


@torch.no_grad()
def capture_subject_states(
    loaded,
    subjects: list[dict],
    layer_index: int,
) -> dict[str, torch.Tensor]:
    """Capture the subject-token state at ``layer_index`` for each subject.

    ``subjects`` rows need ``subject_id`` and ``name``.
    """

    model = loaded.model
    tokenizer = loaded.tokenizer
    device = loaded.device
    states: dict[str, torch.Tensor] = {}
    for subject in subjects:
        prefix = subject_prefix(subject["name"])
        prefix_ids = _encode(tokenizer, prefix, device)
        span_start, span_end = locate_span(
            tokenizer, prefix_ids[0].tolist(), subject["name"], prefix_text=prefix
        )
        captured: dict[str, torch.Tensor] = {}

        def hook(module, args, output):
            hidden = output if torch.is_tensor(output) else output[0]
            captured["state"] = hidden[:, span_end - 1, :].detach().clone()

        handle = model.model.layers[layer_index].register_forward_hook(hook)
        try:
            run_prefix(model, prefix_ids)
        finally:
            handle.remove()
        if "state" not in captured:
            raise RuntimeError(f"failed to capture a subject state for {subject['subject_id']}")
        states[str(subject["subject_id"])] = captured["state"][0]
    return states


def _delta_for_state(
    clean_state: torch.Tensor,
    state: str,
    queried_map: RelationMap,
    other_map: RelationMap,
    alpha: float,
    seed: int,
) -> torch.Tensor:
    if state == "clean":
        return torch.zeros_like(clean_state)
    if state == "relevant":
        return attenuation_delta(clean_state, queried_map, alpha)
    if state == "other":
        return attenuation_delta(clean_state, other_map, alpha)
    if state == "random":
        delta, _ = matched_random_delta(clean_state, queried_map, alpha, seed)
        return delta
    raise ValueError(f"unknown state: {state}")


class RetainedCollector:
    """Collect answer draws and confidence reports under four states."""

    def __init__(self, loaded, settings: CollectorSettings, relation_maps: dict[str, RelationMap]):
        self.loaded = loaded
        self.settings = settings
        self.relation_maps = relation_maps

    def _resume_keys(self, out_path: Path) -> set[tuple]:
        if not out_path.exists():
            return set()
        keys = set()
        with out_path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                keys.add(self._row_key(row))
        return keys

    @staticmethod
    def _row_key(row: dict) -> tuple:
        return (
            str(row.get("subject_id")),
            str(row.get("question_id")),
            str(row.get("state")),
            int(row.get("draw", -1)),
        )

    def collect(
        self,
        pairs: list[dict],
        subject_states: dict[str, torch.Tensor],
        out_path: Path,
        resume: bool = True,
    ) -> dict:
        """Collect every (subject, queried relation, state, draw) cell.

        ``pairs`` rows: subject_id, name, and a ``relations`` list with two
        entries, each ``{relation, question_id, question, reference_answers}``.
        """

        settings = self.settings
        model = self.loaded.model
        tokenizer = self.loaded.tokenizer
        device = self.loaded.device
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done = self._resume_keys(out_path) if resume else set()
        written = 0
        with out_path.open("a") as handle:
            for pair in pairs:
                clean_state = subject_states[str(pair["subject_id"])].to(device)
                prefix = subject_prefix(pair["name"])
                prefix_ids = _encode(tokenizer, prefix, device)
                _, span_end = locate_span(
                    tokenizer, prefix_ids[0].tolist(), pair["name"], prefix_text=prefix
                )
                for index, query in enumerate(pair["relations"]):
                    other = pair["relations"][1 - index]
                    queried_map = self.relation_maps[str(query["relation"])]
                    other_map = self.relation_maps[str(other["relation"])]
                    answer_ids = _encode(tokenizer, answer_suffix(query["question"]), device)
                    confidence_ids = _encode(
                        tokenizer, confidence_suffix(query["question"]), device
                    )
                    for state in STATES:
                        delta = _delta_for_state(
                            clean_state,
                            state,
                            queried_map,
                            other_map,
                            settings.alpha,
                            seed=settings.seed
                            + zlib.crc32(str(pair["subject_id"]).encode()) % 10_000,
                        )
                        transform = additive_transform(
                            (-delta).to(next(model.parameters()).dtype), token_index=span_end - 1
                        )
                        state_cache = run_prefix(
                            model,
                            prefix_ids,
                            layer_index=settings.layer_index,
                            transform=transform,
                        )
                        for draw in range(settings.n_draws):
                            key = (
                                str(pair["subject_id"]),
                                str(query["question_id"]),
                                state,
                                draw,
                            )
                            if key in done:
                                continue
                            generated = generate_branch(
                                model,
                                state_cache,
                                answer_ids,
                                max_new_tokens=settings.max_new_tokens,
                                do_sample=settings.sample_answers,
                                temperature=settings.temperature,
                                top_p=settings.top_p,
                                seed=settings.seed + draw,
                                stop_token_ids=(tokenizer.eos_token_id,),
                            )
                            text = tokenizer.decode(
                                generated[0].tolist(), skip_special_tokens=True
                            ).strip()
                            row = {
                                "kind": "answer",
                                "model": self.loaded.name,
                                "revision": self.loaded.revision,
                                "subject_id": pair["subject_id"],
                                "question_id": query["question_id"],
                                "relation": query["relation"],
                                "queried_relation": query["relation"],
                                "perturbed_relation": state,
                                "state": state,
                                "draw": draw,
                                "layer": settings.layer_index,
                                "alpha": settings.alpha,
                                "perturbation_norm": float(delta.norm()),
                                "answer_text": text,
                                "label": int(match_candidate(text, query["reference_answers"])),
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            }
                            handle.write(json.dumps(row) + "\n")
                            written += 1
                        conf_key = (
                            str(pair["subject_id"]),
                            str(query["question_id"]),
                            state,
                            -1,
                        )
                        if conf_key in done:
                            continue
                        generated_conf = generate_branch(
                            model,
                            state_cache,
                            confidence_ids,
                            max_new_tokens=settings.confidence_tokens,
                            do_sample=False,
                            seed=settings.seed,
                            stop_token_ids=(tokenizer.eos_token_id,),
                        )
                        conf_text = tokenizer.decode(
                            generated_conf[0].tolist(), skip_special_tokens=True
                        ).strip()
                        value, ok = parse_confidence(conf_text)
                        row = {
                            "kind": "confidence",
                            "model": self.loaded.name,
                            "revision": self.loaded.revision,
                            "subject_id": pair["subject_id"],
                            "question_id": query["question_id"],
                            "relation": query["relation"],
                            "queried_relation": query["relation"],
                            "perturbed_relation": state,
                            "state": state,
                            "draw": -1,
                            "layer": settings.layer_index,
                            "alpha": settings.alpha,
                            "perturbation_norm": float(delta.norm()),
                            "confidence_raw": conf_text,
                            "confidence": value,
                            "parse_ok": bool(ok),
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                        handle.write(json.dumps(row) + "\n")
                        written += 1
                    handle.flush()
        return {"path": str(out_path), "rows_written": written}
