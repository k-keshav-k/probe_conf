"""Model and tokenizer loading with pinned revisions and chat templates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import FEWSHOT, SYSTEM_PROMPT


@dataclass
class LoadedModel:
    name: str
    revision: str
    model: object
    tokenizer: object
    device: str
    dtype: str


def load_model_and_tokenizer(
    name: str,
    revision: str | None = None,
    dtype: str = "float16",
    device: str | None = None,
    attn_implementation: str = "sdpa",
) -> LoadedModel:
    """Load a causal LM and tokenizer for one model at a time.

    The caller is responsible for running one model per process on a small
    machine.  Weight revisions are recorded in every output row.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
        dtype
    ]
    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name,
        revision=revision,
        dtype=torch_dtype,
        attn_implementation=attn_implementation,
    ).to(device)
    model.eval()
    resolved_revision = getattr(model.config, "_commit_hash", None) or revision or "unpinned"
    return LoadedModel(
        name=name,
        revision=str(resolved_revision),
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
    )


def chat_prompt_text(tokenizer, question: str, fewshot: tuple = FEWSHOT) -> str:
    """Frozen short-answer prompt: system, few-shot pairs, then the question."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in fewshot:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def chat_prompt_ids(tokenizer, question: str, fewshot: tuple = FEWSHOT) -> list[int]:
    text = chat_prompt_text(tokenizer, question, fewshot)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def confidence_prompt_ids(
    tokenizer,
    question: str,
    *,
    candidate: str | None = None,
    fewshot: tuple = FEWSHOT,
) -> list[int]:
    """Frozen confidence prompt with the same factual context and examples."""

    from ..config import CONFIDENCE_TRACK_A, CONFIDENCE_TRACK_B

    if candidate is None:
        content = CONFIDENCE_TRACK_A.format(question=question)
    else:
        content = CONFIDENCE_TRACK_B.format(question=question, candidate=candidate)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user, assistant in fewshot:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": content})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]
