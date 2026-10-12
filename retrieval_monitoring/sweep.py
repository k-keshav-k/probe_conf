"""S3 development sweep over layer, rank, and alpha.

The first pilot ran at ``alpha=0.5``.  The realized perturbation was about one
percent of the subject-state norm and changed neither success nor confidence,
because the rank-2 map projection of the centered state is small.  This sweep
measures a cheap likelihood proxy on famous subjects -- the teacher-forced mean
log-probability of the reference answer -- so the pilot can run with settings
that have a measurable chance of damaging the queried relation.

The proxy is a development signal only.  The pilot still measures answer
success, and selectivity is what matters: relevant damage must exceed the
other-relation and norm-matched random controls.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import torch

from .collector import (
    _delta_for_state,
    _encode,
    answer_suffix,
    capture_subject_states,
    subject_prefix,
)
from .retained_state import additive_transform, continue_branch, locate_span, run_prefix
from .runner import _cohort_subject_ids, fit_maps_from_source


@dataclass
class SweepConfig:
    layers: tuple[int, ...]
    ranks: tuple[int, ...]
    alphas: tuple[float, ...]
    n_subjects: int = 12
    relations: tuple[str, str] = ("father", "occupation")
    map_subjects: int = 200

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@torch.no_grad()
def reference_score(loaded, prefix_state, suffix_ids: torch.Tensor, reference: str) -> dict:
    """Teacher-forced likelihood of the reference answer from a prefix state."""

    tokenizer = loaded.tokenizer
    ref_ids = tokenizer(" " + str(reference), add_special_tokens=False)["input_ids"]
    if not ref_ids:
        return {"mean_logprob": float("nan"), "first_prob": float("nan")}
    ref_tensor = torch.tensor([ref_ids], dtype=torch.long, device=suffix_ids.device)
    combined = torch.cat([suffix_ids, ref_tensor[:, :-1]], dim=1)
    output = continue_branch(loaded.model, prefix_state, combined)
    start = suffix_ids.shape[1] - 1
    logits = output.logits[0, start : start + len(ref_ids)].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    gathered = logprobs.gather(1, ref_tensor[0].unsqueeze(1)).squeeze(1)
    probs = torch.softmax(logits[0], dim=-1)
    return {
        "mean_logprob": float(gathered.mean()),
        "first_prob": float(probs[ref_ids[0]]),
    }


def _summarize(layer: int, rank: int, alpha: float, records: list[dict]) -> dict:
    def mean(key: str) -> float:
        values = [record[key] for record in records]
        return float(torch.tensor(values, dtype=torch.float64).mean()) if values else float("nan")

    clean_norm = mean("clean_norm")
    return {
        "layer": layer,
        "rank": rank,
        "alpha": alpha,
        "n_records": len(records),
        "clean_mean_logprob": mean("clean_mean_logprob"),
        "clean_first_prob": mean("clean_first_prob"),
        "damage_relevant": mean("delta_relevant"),
        "damage_other": mean("delta_other"),
        "damage_random": mean("delta_random"),
        "selectivity_vs_random": mean("delta_relevant") - mean("delta_random"),
        "selectivity_vs_other": mean("delta_relevant") - mean("delta_other"),
        "delta_norm": mean("delta_norm"),
        "delta_fraction": mean("delta_norm") / max(clean_norm, 1e-9),
        "records": records,
    }


def run_sweep(loaded, source, pairs: list[dict], config: SweepConfig, *, run_dir=None, out_dir=None) -> dict:
    """Measure the reference-likelihood damage for every setting.

    ``source`` is the normalized cohort table used to fit the relation maps.
    ``pairs`` rows need ``subject_id``, ``name``, and ``relations``.
    """

    pilot_ids = {str(pair["subject_id"]) for pair in pairs}
    exclude = pilot_ids | _cohort_subject_ids(run_dir)
    pilot_subjects = [{"subject_id": pair["subject_id"], "name": pair["name"]} for pair in pairs]
    device = loaded.device
    dtype = next(loaded.model.parameters()).dtype

    settings: list[dict] = []
    for layer in config.layers:
        states = capture_subject_states(loaded, pilot_subjects, layer)
        for rank in config.ranks:
            maps = fit_maps_from_source(
                loaded,
                source,
                config.relations,
                layer,
                rank=rank,
                per_relation=config.map_subjects,
                exclude=exclude,
            )
            for alpha in config.alphas:
                records: list[dict] = []
                for pair in pairs:
                    clean_state = states[str(pair["subject_id"])].to(device)
                    prefix = subject_prefix(pair["name"])
                    prefix_ids = _encode(loaded.tokenizer, prefix, device)
                    _, span_end = locate_span(
                        loaded.tokenizer, prefix_ids[0].tolist(), pair["name"], prefix_text=prefix
                    )
                    seed = zlib.crc32(str(pair["subject_id"]).encode()) % 10_000
                    for index, query in enumerate(pair["relations"]):
                        other = pair["relations"][1 - index]
                        queried_map = maps[str(query["relation"])]
                        other_map = maps[str(other["relation"])]
                        suffix_ids = _encode(
                            loaded.tokenizer, answer_suffix(query["relation"]), device
                        )
                        scores: dict[str, dict] = {}
                        norms: dict[str, float] = {}
                        for state in ("clean", "relevant", "other", "random"):
                            delta = _delta_for_state(
                                clean_state, state, queried_map, other_map, alpha, seed
                            )
                            prefix_state = run_prefix(
                                loaded.model,
                                prefix_ids,
                                layer_index=layer,
                                transform=additive_transform(
                                    (-delta).to(dtype), token_index=span_end - 1
                                ),
                            )
                            scores[state] = reference_score(
                                loaded,
                                prefix_state,
                                suffix_ids,
                                query["reference_answers"][0],
                            )
                            norms[state] = float(delta.norm())
                        records.append(
                            {
                                "subject_id": str(pair["subject_id"]),
                                "relation": str(query["relation"]),
                                "clean_mean_logprob": scores["clean"]["mean_logprob"],
                                "clean_first_prob": scores["clean"]["first_prob"],
                                "delta_relevant": scores["clean"]["mean_logprob"]
                                - scores["relevant"]["mean_logprob"],
                                "delta_other": scores["clean"]["mean_logprob"]
                                - scores["other"]["mean_logprob"],
                                "delta_random": scores["clean"]["mean_logprob"]
                                - scores["random"]["mean_logprob"],
                                "delta_norm": norms["relevant"],
                                "clean_norm": float(clean_state.norm()),
                            }
                        )
                settings.append(_summarize(layer, rank, alpha, records))

    report = {"config": config.as_dict(), "settings": settings}
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "sweep_report.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        report["path"] = str(path)
    return report
