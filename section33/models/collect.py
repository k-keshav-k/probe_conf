"""Collection for the two tracks.

Track A (before-answer): one greedy answer per question, one label, one
confidence report, and reference-free features at the last prompt token.

Track B (after-answer): eight sampled answers per question, one label per
unique candidate, one confidence report per unique candidate, and features at
the last content token of the candidate.  Duplicate candidates keep their
sampling multiplicity.

Reference answers live in the labels interface only.  Feature rows come from
``FeatureRow`` and contain token IDs, positions, and reference-free features.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..config import PROTOCOL, STOP_STRINGS
from ..data.labels import label_policy_hash, match_candidate
from ..features import families as fam
from ..features import mlp_math as mm
from ..features import probability as prob
from ..features.hidden import HiddenProjector
from ..schema import AnswerRecord, ConfidenceRecord, FeatureRow
from ..utils import append_jsonl, batched, log, read_jsonl, set_seed, sha256_text
from .contract import MLPCapture, get_layers, mlp_modules
from .loader import LoadedModel, chat_prompt_ids, confidence_prompt_ids

FEATURE_SPEC_VERSION = "section33_features_v1"
REFUSAL_PATTERNS = (
    "i don't know",
    "i do not know",
    "unknown",
    "not sure",
    "cannot answer",
    "can't answer",
    "no answer",
)


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------
def parse_confidence(text: str) -> tuple[float | None, bool]:
    """Strict parse: only the requested integer in ``[0, 100]``."""

    cleaned = text.strip()
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()
    if re.fullmatch(r"\d{1,3}", cleaned):
        value = int(cleaned)
        if 0 <= value <= 100:
            return float(value), True
    return None, False


def stop_token_ids(tokenizer, stops: tuple = STOP_STRINGS) -> set[int]:
    ids = set()
    for stop in stops:
        encoded = tokenizer(stop, add_special_tokens=False)["input_ids"]
        if len(encoded) == 1:
            ids.add(encoded[0])
    return ids


def trim_at_stop(ids: list[int], stop_ids: set[int]) -> tuple[list[int], bool]:
    for index, token in enumerate(ids):
        if token in stop_ids:
            return ids[:index], False
    return ids, True


def is_refusal(text: str) -> bool:
    lowered = text.strip().lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def model_slug(name: str) -> str:
    return name.replace("/", "__").replace("-", "_").lower()


def candidate_key(model: str, revision: str, token_ids: list[int]) -> str:
    """Stable candidate identity shared by the labels, confidence, and feature rows.

    Includes the model and revision so rows from different runs can never merge,
    and the exact token IDs so two candidates that decode to the same text stay
    distinct.
    """

    return sha256_text(json.dumps([model, revision, list(token_ids)]))[:16]


def _question_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["question_id"] for row in read_jsonl(path)}


def _drop_incomplete_questions(path: Path, keep: set[str]) -> int:
    """Rewrite a jsonl file without rows for questions missing from any interface."""

    if not path.exists():
        return 0
    rows = read_jsonl(path)
    kept = [row for row in rows if row["question_id"] in keep]
    if len(kept) == len(rows):
        return 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        for row in kept:
            handle.write(json.dumps(row) + "\n")
    os.replace(tmp, path)
    return len(rows) - len(kept)


# ---------------------------------------------------------------------------
# Cached float32 weights
# ---------------------------------------------------------------------------
class ModelWeights:
    """Float32 MLP weights and value-column norms for the frozen equations."""

    def __init__(self, model, device: str):
        self.device = device
        self.w_gate: list[torch.Tensor] = []
        self.w_up: list[torch.Tensor] = []
        self.w_down: list[torch.Tensor] = []
        self.v_norms: list[torch.Tensor] = []
        for layer in get_layers(model):
            gate, up, down = mlp_modules(layer)
            self.w_gate.append(gate.weight.detach().to(device, torch.float32))
            self.w_up.append(up.weight.detach().to(device, torch.float32))
            self.w_down.append(down.weight.detach().to(device, torch.float32))
            self.v_norms.append(down.weight.detach().float().norm(dim=0))
        self.n_layers = len(self.w_gate)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
@dataclass
class CollectSettings:
    batch_questions: int = 4
    max_content_tokens: int = PROTOCOL.content_token_limit
    generation_seed: int = 0
    track_b_samples: int = PROTOCOL.track_b_samples
    families: tuple[str, ...] = PROTOCOL.families
    positions: tuple[str, ...] = ("last_prompt", "answer_last_content")
    max_questions: int | None = None
    resume: bool = True
    tag: str = ""  # shard or run tag appended to output file names
    hidden_proj: bool = True
    hidden_proj_dim: int = 64
    hidden_proj_seed: int = 11


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def _generate(
    loaded: LoadedModel,
    sequences: list[list[int]],
    max_new_tokens: int,
    do_sample: bool,
    seed: int,
) -> list[list[tuple[list[int], str]]]:
    """Generate ``n`` continuations for each sequence.

    ``sequences`` already contains each prompt repeated ``n`` times.  Returns a
    nested list: one entry per sequence, each with ``(new_ids, text)`` pairs.
    """

    tokenizer = loaded.tokenizer
    pad_id = tokenizer.pad_token_id
    max_len = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, ids in enumerate(sequences):
        input_ids[row, max_len - len(ids) :] = torch.tensor(ids, dtype=torch.long)
        attention[row, max_len - len(ids) :] = 1
    attention = attention.to(loaded.device)
    input_ids = input_ids.to(loaded.device)

    set_seed(seed)
    output = loaded.model.generate(
        input_ids=input_ids,
        attention_mask=attention,
        do_sample=do_sample,
        temperature=PROTOCOL.temperature if do_sample else None,
        top_p=PROTOCOL.top_p if do_sample else None,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
    )
    new_tokens = output[:, max_len:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    stop_ids = stop_token_ids(tokenizer)
    results: list[list[tuple[list[int], str]]] = []
    for row, text in enumerate(decoded):
        ids = new_tokens[row].tolist()
        ids = [t for t in ids if t != pad_id]
        trimmed, hit_limit = trim_at_stop(ids, stop_ids)
        results.append([(trimmed, text.split("\n")[0].strip(), hit_limit)])
    return results


def _repeat_for_samples(prompts: list[list[int]], n_samples: int) -> tuple[list[list[int]], list[int]]:
    sequences: list[list[int]] = []
    owner: list[int] = []
    for index, prompt in enumerate(prompts):
        for _ in range(n_samples):
            sequences.append(prompt)
            owner.append(index)
    return sequences, owner


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features_for_sequences(
    loaded: LoadedModel,
    weights: ModelWeights,
    sequences: list[list[int]],
    capture_positions: list[list[int]],
    position_names: list[str],
    families: tuple[str, ...] = PROTOCOL.families,
    hidden_projector: "HiddenProjector | None" = None,
    probability: bool = True,
    sequence_spans: list[tuple[int, int] | None] | None = None,
) -> list[dict[str, float]]:
    """Reference-free features for every sequence and capture position.

    ``capture_positions[i]`` holds absolute token indices into ``sequences[i]``.
    ``position_names[p]`` names capture slot ``p`` for every row.

    The function adds:

    - the frozen MLP families,
    - fixed random projections of the pre-block and post-block hidden states
      (``hiddenproj__...``) when a projector is given,
    - probability features from the model's own logits (``prob__...``),
    - sequence log-probability statistics for Track B when spans are given.
    """

    tokenizer = loaded.tokenizer
    pad_id = tokenizer.pad_token_id
    max_len = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros((len(sequences), max_len), dtype=torch.long)
    for row, ids in enumerate(sequences):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[row, : len(ids)] = 1
    attention = attention.to(loaded.device)
    input_ids = input_ids.to(loaded.device)

    capture = MLPCapture(loaded.model, capture_positions, keep_device=True, keys=("x", "a", "b", "s"))
    capture.register()
    try:
        outputs = loaded.model(
            input_ids=input_ids,
            attention_mask=attention,
            output_hidden_states=True,
            use_cache=False,
        )
    finally:
        capture.remove()

    hidden = outputs.hidden_states  # tuple of (B, S, d)
    logits = outputs.logits  # (B, S, V)
    rows = torch.arange(len(sequences), device=loaded.device)
    columns = torch.as_tensor(capture_positions, device=loaded.device, dtype=torch.long)
    n_positions = len(position_names)
    results: list[dict[str, float]] = [dict() for _ in sequences]

    for layer in range(weights.n_layers):
        x = capture.captured[layer]["x"].float()
        a = capture.captured[layer]["a"].float()
        b = capture.captured[layer]["b"].float()
        s = capture.captured[layer]["s"].float()
        w_down = weights.w_down[layer]
        o = torch.einsum("bpf,df->bpd", s, w_down)
        w2_dot = torch.einsum("bpd,df->bpf", x, w_down)
        h_pre = hidden[layer][rows.unsqueeze(1), columns]
        h_post = hidden[layer + 1][rows.unsqueeze(1), columns]

        # Build all families for the whole batch of positions at once, then
        # move one tensor to the CPU per statistic.  The per-token scalar path
        # caused thousands of host-device syncs per sequence.
        batch = x.shape[0]
        flat = batch * n_positions
        tensors = mm.MLPTensors(
            x=x.reshape(flat, -1),
            h_pre=h_pre.reshape(flat, -1),
            h_post=h_post.reshape(flat, -1),
            a=a.reshape(flat, -1),
            b=b.reshape(flat, -1),
            s=s.reshape(flat, -1),
            o=o.reshape(flat, -1),
            v_norms=weights.v_norms[layer],
            w2_dot=w2_dot.reshape(flat, -1),
        )
        stat_tensors = fam.build_all_tensors(tensors, families)
        stat_names = list(stat_tensors)
        stacked = torch.stack([stat_tensors[name] for name in stat_names]).detach().cpu()
        stat_values = dict(zip(stat_names, stacked.tolist()))
        for slot, position in enumerate(position_names):
            for row in range(batch):
                index = row * n_positions + slot
                target = results[row]
                for name, values in stat_values.items():
                    target[f"{name}__{position}__l{layer}"] = values[index]

        if hidden_projector is not None and layer in hidden_projector.layers:
            pre_projection = hidden_projector.project(h_pre, layer, "pre").detach().cpu()
            post_projection = hidden_projector.project(h_post, layer, "post").detach().cpu()
            for row in range(batch):
                for slot, position in enumerate(position_names):
                    for kind, projection in (("pre", pre_projection), ("post", post_projection)):
                        values = projection[row, slot].tolist()
                        for index, value in enumerate(values):
                            results[row][
                                f"hiddenproj__{kind}__{position}__l{layer}__k{index}"
                            ] = value

    if probability:
        position_logits = logits[rows.unsqueeze(1), columns]  # (B, P, V)
        for row in range(len(sequences)):
            for slot in range(n_positions):
                results[row].update(
                    prob.next_token_features(position_logits[row, slot], position_names[slot])
                )
            if sequence_spans is not None and sequence_spans[row] is not None:
                start, end = sequence_spans[row]
                if end - start >= 1:
                    results[row].update(
                        prob.sequence_features(
                            logits[row, start - 1 : end - 1],
                            sequences[row][start:end],
                            position_names[0],
                        )
                    )
    return results


# ---------------------------------------------------------------------------
# Track A
# ---------------------------------------------------------------------------
def collect_track_a(
    loaded: LoadedModel,
    questions: list,
    settings: CollectSettings,
    paths: dict[str, Path],
) -> dict:
    slug = model_slug(loaded.name)
    suffix = f"__{slug}" + (f"__{settings.tag}" if settings.tag else "")
    labels_path = paths["labels"] / f"trackA_answers{suffix}.jsonl"
    conf_path = paths["labels"] / f"trackA_confidence{suffix}.jsonl"
    feat_path = paths["features"] / f"trackA_features{suffix}.jsonl"
    for path in (labels_path, conf_path, feat_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if settings.resume:
        done = _question_ids(labels_path) & _question_ids(conf_path) & _question_ids(feat_path)
        for path in (labels_path, conf_path, feat_path):
            _drop_incomplete_questions(path, done)
    todo = [q for q in questions if q.question_id not in done]
    if settings.max_questions:
        todo = todo[: settings.max_questions]
    log(f"track A: {len(done)} done, {len(todo)} to collect")

    weights = ModelWeights(loaded.model, loaded.device)
    projector = (
        HiddenProjector(
            weights.n_layers,
            loaded.model.config.hidden_size,
            dim=settings.hidden_proj_dim,
            seed=settings.hidden_proj_seed,
        )
        if settings.hidden_proj
        else None
    )

    for batch_index, batch in enumerate(batched(todo, settings.batch_questions)):
        prompts = [chat_prompt_ids(loaded.tokenizer, q.question) for q in batch]
        sequences, owner = _repeat_for_samples(prompts, 1)
        generated = _generate(
            loaded, sequences, settings.max_content_tokens, do_sample=False,
            seed=settings.generation_seed + batch_index,
        )
        generated = [generated[i][0] for i in range(len(generated))]  # one sample each

        answer_rows, confidence_rows, feature_rows = [], [], []
        for q_index, (question, (answer_ids, answer_text, hit_limit)) in enumerate(zip(batch, generated)):
            key = candidate_key(loaded.name, loaded.revision, answer_ids)
            label = match_candidate(answer_text, question.reference_answers)
            answer_rows.append(
                AnswerRecord(
                    model=loaded.name, revision=loaded.revision,
                    subject_id=question.subject_id, question_id=question.question_id,
                    split=question.split, track="A", answer_index=0,
                    generation_seed=settings.generation_seed + batch_index,
                    candidate_text=answer_text, candidate_key=key, label=label,
                    label_policy_version=label_policy_hash(),
                    reference_answers=list(question.reference_answers),
                    empty=len(answer_ids) == 0, truncated=hit_limit,
                    refusal=is_refusal(answer_text),
                    relation=getattr(question, "relation", ""),
                    popularity=getattr(question, "popularity", None),
                    stratum=getattr(question, "stratum", ""),
                ).to_row()
            )

        feature_values = extract_features_for_sequences(
            loaded, weights,
            sequences=prompts,  # prompt only: no generated token enters Track A features
            capture_positions=[[len(prompt) - 1] for prompt in prompts],
            position_names=["last_prompt"],
            families=settings.families,
            hidden_projector=projector,
            probability=True,
        )
        for q_index, (question, (answer_ids, answer_text, _), feature_dict) in enumerate(zip(batch, generated, feature_values)):
            feature_rows.append(
                FeatureRow(
                    model=loaded.name, revision=loaded.revision,
                    subject_id=question.subject_id, question_id=question.question_id,
                    split=question.split, track="A", answer_index=0,
                    generation_seed=settings.generation_seed + batch_index,
                    position="last_prompt", features=feature_dict,
                    prompt_token_hash=sha256_text(json.dumps(prompts[q_index]))[:16],
                    candidate_token_ids=answer_ids,
                    candidate_key=candidate_key(loaded.name, loaded.revision, answer_ids),
                    content_length=len(answer_ids),
                    feature_spec_version=FEATURE_SPEC_VERSION,
                ).to_row()
            )

        conf_prompts = [confidence_prompt_ids(loaded.tokenizer, q.question) for q in batch]
        conf_sequences, _ = _repeat_for_samples(conf_prompts, 1)
        conf_generated = _generate(loaded, conf_sequences, 8, do_sample=False,
                                   seed=settings.generation_seed + batch_index)
        for question, (answer_ids, _, _), entry in zip(batch, generated, conf_generated):
            text = entry[0][1]
            value, ok = parse_confidence(text)
            confidence_rows.append(
                ConfidenceRecord(
                    model=loaded.name, revision=loaded.revision,
                    subject_id=question.subject_id, question_id=question.question_id,
                    split=question.split, track="A", answer_index=0,
                    candidate_key=candidate_key(loaded.name, loaded.revision, answer_ids),
                    generation_seed=settings.generation_seed + batch_index,
                    prompt_version="section33_prompts_v1",
                    raw_text=text, value=value, parse_ok=ok,
                ).to_row()
            )

        append_jsonl(labels_path, answer_rows)
        append_jsonl(conf_path, confidence_rows)
        append_jsonl(feat_path, feature_rows)
        if batch_index % 10 == 0:
            log(f"track A batch {batch_index}: {len(answer_rows)} questions")

    return {"labels": str(labels_path), "confidence": str(conf_path), "features": str(feat_path)}


# ---------------------------------------------------------------------------
# Track B
# ---------------------------------------------------------------------------
def collect_track_b(
    loaded: LoadedModel,
    questions: list,
    settings: CollectSettings,
    paths: dict[str, Path],
) -> dict:
    slug = model_slug(loaded.name)
    suffix = f"__{slug}" + (f"__{settings.tag}" if settings.tag else "")
    labels_path = paths["labels"] / f"trackB_answers{suffix}.jsonl"
    conf_path = paths["labels"] / f"trackB_confidence{suffix}.jsonl"
    feat_path = paths["features"] / f"trackB_features{suffix}.jsonl"
    for path in (labels_path, conf_path, feat_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if settings.resume:
        done = _question_ids(labels_path) & _question_ids(conf_path) & _question_ids(feat_path)
        for path in (labels_path, conf_path, feat_path):
            _drop_incomplete_questions(path, done)
    todo = [q for q in questions if q.question_id not in done]
    if settings.max_questions:
        todo = todo[: settings.max_questions]
    log(f"track B: {len(done)} done, {len(todo)} to collect")

    weights = ModelWeights(loaded.model, loaded.device)
    projector = (
        HiddenProjector(
            weights.n_layers,
            loaded.model.config.hidden_size,
            dim=settings.hidden_proj_dim,
            seed=settings.hidden_proj_seed,
        )
        if settings.hidden_proj
        else None
    )

    for batch_index, batch in enumerate(batched(todo, settings.batch_questions)):
        prompts = [chat_prompt_ids(loaded.tokenizer, q.question) for q in batch]
        sequences, _ = _repeat_for_samples(prompts, settings.track_b_samples)
        generated = _generate(
            loaded, sequences, settings.max_content_tokens, do_sample=True,
            seed=settings.generation_seed + batch_index,
        )
        # Regroup samples per question.
        per_question: list[list[tuple[list[int], str, bool]]] = [[] for _ in batch]
        for index, entries in enumerate(generated):
            per_question[index // settings.track_b_samples].extend(entries)

        # Unique candidates per question, with multiplicity.
        replay_sequences: list[list[int]] = []
        replay_meta: list[tuple[int, dict]] = []
        for q_index, samples in enumerate(per_question):
            seen: dict[tuple[int, ...], dict] = {}
            for answer_ids, text, hit_limit in samples:
                key = tuple(answer_ids)
                if key not in seen:
                    seen[key] = {"ids": answer_ids, "text": text, "multiplicity": 0, "truncated": hit_limit}
                seen[key]["multiplicity"] += 1
            for item in seen.values():
                replay_sequences.append(prompts[q_index] + item["ids"])
                replay_meta.append((q_index, item))

        capture_positions = [
            [len(sequence) - 1, len(prompts[q_index]) - 1]
            for sequence, (q_index, _) in zip(replay_sequences, replay_meta)
        ]
        sequence_spans = [
            (len(prompts[q_index]), len(sequence)) for sequence, (q_index, _) in zip(replay_sequences, replay_meta)
        ]
        feature_values = extract_features_for_sequences(
            loaded, weights, replay_sequences, capture_positions,
            position_names=["answer_last_content", "last_prompt"],
            families=settings.families,
            hidden_projector=projector,
            probability=True,
            sequence_spans=sequence_spans,
        )

        answer_rows, confidence_rows, feature_rows = [], [], []
        conf_prompts, conf_meta = [], []
        for meta_index, ((q_index, item), feature_dict) in enumerate(zip(replay_meta, feature_values)):
            question = batch[q_index]
            key = candidate_key(loaded.name, loaded.revision, item["ids"])
            label = match_candidate(item["text"], question.reference_answers)
            answer_rows.append(
                AnswerRecord(
                    model=loaded.name, revision=loaded.revision,
                    subject_id=question.subject_id, question_id=question.question_id,
                    split=question.split, track="B", answer_index=meta_index,
                    generation_seed=settings.generation_seed + batch_index,
                    candidate_text=item["text"], candidate_key=key, label=label,
                    label_policy_version=label_policy_hash(),
                    reference_answers=list(question.reference_answers),
                    empty=len(item["ids"]) == 0, truncated=item["truncated"],
                    refusal=is_refusal(item["text"]),
                    multiplicity=item["multiplicity"],
                    relation=getattr(question, "relation", ""),
                    popularity=getattr(question, "popularity", None),
                    stratum=getattr(question, "stratum", ""),
                ).to_row()
            )
            features = feature_dict
            for position in ("answer_last_content", "last_prompt"):
                feature_rows.append(
                    FeatureRow(
                        model=loaded.name, revision=loaded.revision,
                        subject_id=question.subject_id, question_id=question.question_id,
                        split=question.split, track="B", answer_index=meta_index,
                        generation_seed=settings.generation_seed + batch_index,
                        position=position, features=features,
                        prompt_token_hash=sha256_text(json.dumps(prompts[q_index]))[:16],
                        candidate_token_ids=item["ids"],
                        candidate_key=key,
                        content_length=len(item["ids"]),
                        feature_spec_version=FEATURE_SPEC_VERSION,
                        multiplicity=item["multiplicity"],
                    ).to_row()
                )
            conf_prompts.append(confidence_prompt_ids(loaded.tokenizer, question.question, candidate=item["text"]))
            conf_meta.append((question, meta_index, item))

        conf_sequences = conf_prompts
        conf_generated = _generate(loaded, conf_sequences, 8, do_sample=False,
                                   seed=settings.generation_seed + batch_index)
        for entry, (question, meta_index, item) in zip(conf_generated, conf_meta):
            text = entry[0][1]
            value, ok = parse_confidence(text)
            confidence_rows.append(
                ConfidenceRecord(
                    model=loaded.name, revision=loaded.revision,
                    subject_id=question.subject_id, question_id=question.question_id,
                    split=question.split, track="B", answer_index=meta_index,
                    candidate_key=candidate_key(loaded.name, loaded.revision, item["ids"]),
                    generation_seed=settings.generation_seed + batch_index,
                    prompt_version="section33_prompts_v1",
                    raw_text=text, value=value, parse_ok=ok,
                ).to_row()
            )

        append_jsonl(labels_path, answer_rows)
        append_jsonl(conf_path, confidence_rows)
        append_jsonl(feat_path, feature_rows)
        if batch_index % 10 == 0:
            log(f"track B batch {batch_index}: {len(answer_rows)} unique candidates")

    return {"labels": str(labels_path), "confidence": str(conf_path), "features": str(feat_path)}


def collect(
    loaded: LoadedModel,
    questions: list,
    settings: CollectSettings,
    paths: dict[str, Path],
    track: str,
) -> dict:
    if track.upper() == "A":
        return collect_track_a(loaded, questions, settings, paths)
    if track.upper() == "B":
        return collect_track_b(loaded, questions, settings, paths)
    raise ValueError(f"unknown track: {track}")
