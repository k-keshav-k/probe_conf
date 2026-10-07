"""Collect correctness labels, verbalized confidence, and internal features.

Pipeline per question (batched):
  1. sample K answers at T=1.0 -> fraction correct (consistency score)
  2. greedy answer -> greedy_correct
  3. separate confidence prompt -> confidence 1-100
  4. one forward pass over `prompt + gold answer` with hooks:
       - residual hidden states at last-prompt and answer token positions
       - MLP gate pre-activations (`gate_proj` out), gated values (`down_proj` in),
         MLP writes (`down_proj` out) at the same positions
       - next-token logits at the last prompt token (p_gold, entropy, top-1 margin)
       - answer-direction alignment of the MLP write: W2-column projections onto
         the gold/competitor unembedding rows (batched, GPU)
     -> summary statistics dataframe + raw arrays (npz per batch)

Usage:
    python scripts/collect.py --k 10 --limit 100        # quick pass
    python scripts/collect.py --k 30                    # full pilot
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import (  # noqa: E402
    CONFIDENCE_PROMPT,
    MAX_NEW_TOKENS,
    MODEL_NAME,
    N_COMPETITORS,
    PERSON_RELATIONS,
    RAW_DIR,
    RESULTS_DIR,
    SYSTEM_PROMPT,
    TEMPERATURE,
    TOP_P,
    Question,
    first_line,
    load_questions,
    matches_any_alias,
)

SUMMARY_STATS = [
    "ent_gate",
    "ent_gate_sq",
    "active_frac",
    "top5_gate",
    "gini_gate",
    "max_gate",
    "kurt_gate",
    "ent_sel_sq",
    "l2_sel",
    "l2_down",
    "down_norm_ratio",
    "cos_down_h",
    "cos_sel_agold",
    "proj_gold_norm",
    "mlp_margin",
    "ent_cand",
    "logit_mlp_gold",
]
POSITIONS = ("last", "ans")
EPS = 1e-12


# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise AttributeError("cannot locate transformer layers")


def build_prompts(tokenizer, questions: list[Question], fewshot: bool = False) -> None:
    fewshot_turns = [
        ("Who is the father of George W. Bush?", "George H. W. Bush"),
        ("Who is the mother of Prince William?", "Diana, Princess of Wales"),
        ("In what city was Albert Einstein born?", "Ulm"),
        ("What is Barack Obama's occupation?", "politician"),
    ]
    for q in questions:
        if q.prompt:
            continue
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if fewshot:
            for u, a in fewshot_turns:
                messages.append({"role": "user", "content": u})
                messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": q.question})
        q.prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def parse_confidence(text: str) -> float:
    import re

    m = re.search(r"\b(\d{1,3})\b", text)
    if not m:
        return -1.0
    v = int(m.group(1))
    return float(v) if 1 <= v <= 100 else -1.0


# ---------------------------------------------------------------------------
# vectorized numpy stats (operate on the last axis)
# ---------------------------------------------------------------------------
def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def entropy_logits_np(x, axis=-1):
    p = softmax_np(x, axis=axis)
    return -np.sum(p * np.log(p + EPS), axis=axis)


def entropy_sq_np(x, axis=-1):
    q = x * x
    return entropy_logits_np(np.log(q + EPS), axis=axis)


def gini_np(x, axis=-1):
    xs = np.sort(np.abs(x), axis=axis)
    n = xs.shape[axis]
    cum = np.cumsum(xs, axis=axis)
    denom = cum[..., -1:]
    return (n + 1 - 2 * np.sum(cum / (denom + EPS), axis=axis)) / n


def topk_mass_np(x, k, axis=-1):
    xs = np.sort(x, axis=axis)
    return np.sum(xs[..., -k:], axis=axis) / (np.sum(x, axis=axis) + EPS)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_correctness(
    model, tokenizer, questions, k, batch_questions, device
):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    n = len(questions)
    frac = np.zeros(n, dtype=np.float32)
    greedy_ok = np.zeros(n, dtype=np.float32)
    samples: list[list[str]] = [[] for _ in range(n)]

    t0 = time.time()
    for start in range(0, n, batch_questions):
        batch = questions[start : start + batch_questions]
        texts = [q.prompt for q in batch for _ in range(k)]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded = tokenizer.batch_decode(
            out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for j, q in enumerate(batch):
            chunk = decoded[j * k : (j + 1) * k]
            ok = [matches_any_alias(first_line(c), q.aliases) for c in chunk]
            frac[start + j] = float(np.mean(ok))
            samples[start + j] = chunk

        enc1 = tokenizer(
            [q.prompt for q in batch],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        out1 = model.generate(
            **enc1, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
        dec1 = tokenizer.batch_decode(
            out1[:, enc1["input_ids"].shape[1] :], skip_special_tokens=True
        )
        for j, q in enumerate(batch):
            greedy_ok[start + j] = float(matches_any_alias(first_line(dec1[j]), q.aliases))

        done = min(start + batch_questions, n)
        if (start // batch_questions) % 5 == 0 or done == n:
            rate = done / max(1e-9, time.time() - t0)
            print(
                f"[gen] {done}/{n} questions ({rate:.2f} q/s), "
                f"mean frac_correct so far {frac[:done].mean():.3f}",
                flush=True,
            )
    return frac, greedy_ok, samples


@torch.no_grad()
def generate_confidence(model, tokenizer, questions, batch_questions, device):
    tokenizer.padding_side = "left"
    out_vals = np.full(len(questions), -1.0, dtype=np.float32)
    for start in range(0, len(questions), batch_questions):
        batch = questions[start : start + batch_questions]
        texts = []
        for q in batch:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": CONFIDENCE_PROMPT.format(question=q.question)},
            ]
            texts.append(
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(
            **enc, do_sample=False, max_new_tokens=8, pad_token_id=tokenizer.pad_token_id
        )[:, enc["input_ids"].shape[1] :]
        for j, text in enumerate(tokenizer.batch_decode(gen, skip_special_tokens=True)):
            out_vals[start + j] = parse_confidence(first_line(text))
    return out_vals


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------
class HookCapture:
    """Capture gate_proj / down_proj tensors at two token positions per row."""

    def __init__(self, layers, prompt_lens: list[int]):
        self.layers = layers
        self.positions = np.array([[lp - 1, lp] for lp in prompt_lens], dtype=np.int64)
        self.captured: dict[int, dict[str, torch.Tensor]] = {}
        self.handles: list = []

    def _index(self, tensor, batch_size):
        pos = torch.as_tensor(self.positions, device=tensor.device).reshape(-1)
        idx_r = torch.arange(batch_size, device=tensor.device).repeat_interleave(2)
        return tensor[idx_r, pos].reshape(batch_size, 2, -1)

    def register(self):
        for li, layer in enumerate(self.layers):
            mlp = layer.mlp
            self.captured[li] = {}

            def gate_hook(module, inputs, output, li=li):
                self.captured[li]["gate_pre"] = self._index(output, output.shape[0]).to("cpu", torch.float16)

            def down_pre_hook(module, inputs, li=li):
                sel = inputs[0]
                self.captured[li]["sel"] = self._index(sel, sel.shape[0]).to("cpu", torch.float16)

            def down_hook(module, inputs, output, li=li):
                self.captured[li]["down"] = self._index(output, output.shape[0]).to("cpu", torch.float16)

            self.handles.append(mlp.gate_proj.register_forward_hook(gate_hook))
            self.handles.append(mlp.down_proj.register_forward_pre_hook(down_pre_hook))
            self.handles.append(mlp.down_proj.register_forward_hook(down_hook))

    def stack(self, key: str) -> torch.Tensor:
        return torch.stack([self.captured[l][key] for l in range(len(self.layers))], dim=1)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


# ---------------------------------------------------------------------------
# feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(
    model,
    tokenizer,
    questions,
    batch_questions,
    device,
    run_tag,
    save_raw=True,
    resume=False,
):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    layers = get_layers(model)
    L = len(layers)
    D_gpu = model.get_output_embeddings().weight.detach().float()  # (V, d)
    V, d_model = D_gpu.shape
    d_ff = model.config.intermediate_size
    W2_gpu = [layer.mlp.down_proj.weight.detach().float() for layer in layers]  # (d, d_ff)

    partial_path = RESULTS_DIR / f"{run_tag}_features_partial.jsonl"
    rows: list[dict] = []
    done_batches: set[int] = set()
    if resume and partial_path.exists():
        with open(partial_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        done_ids = {r["id"] for r in rows}
        for start in range(0, len(questions), batch_questions):
            f = RAW_DIR / f"{run_tag}_batch{start:05d}.npz"
            if f.exists():
                ids = {int(i) for i in np.load(f)["ids"]}
                if ids <= done_ids:
                    done_batches.add(start)
        print(
            f"[feat] resuming: {len(done_batches)} batches done, "
            f"{len(rows)} rows loaded, {len(questions)} questions total",
            flush=True,
        )
    out_mode = "a" if (resume and partial_path.exists()) else "w"
    fout = open(partial_path, out_mode)
    n = len(questions)

    for start in range(0, n, batch_questions):
        if start in done_batches:
            continue
        batch = questions[start : start + batch_questions]
        B = len(batch)

        prompt_ids = [tokenizer(q.prompt, add_special_tokens=False)["input_ids"] for q in batch]
        ans_ids = [tokenizer(" " + q.obj, add_special_tokens=False)["input_ids"][:1] for q in batch]
        full_ids = [p + a for p, a in zip(prompt_ids, ans_ids)]
        prompt_lens = [len(p) for p in prompt_ids]
        seq_len = max(len(f) for f in full_ids)

        input_ids = torch.full((B, seq_len), pad_id, dtype=torch.long)
        attn = torch.zeros((B, seq_len), dtype=torch.long)
        for i, f in enumerate(full_ids):
            input_ids[i, : len(f)] = torch.tensor(f, dtype=torch.long)
            attn[i, : len(f)] = 1
        input_ids, attn = input_ids.to(device), attn.to(device)

        cap = HookCapture(layers, prompt_lens)
        cap.register()
        outputs = model(
            input_ids=input_ids, attention_mask=attn,
            output_hidden_states=True, use_cache=False,
        )
        cap.remove()

        # hidden states (L+1, B, 2, d) -> cpu fp16
        hs = torch.stack(outputs.hidden_states, dim=0)
        pos = torch.as_tensor(cap.positions, device=device).reshape(-1)
        idx_r = torch.arange(B, device=device).repeat_interleave(2)
        h_pos = hs[:, idx_r, pos].reshape(hs.shape[0], B, 2, -1).to("cpu", torch.float16)
        h_last, h_ans = h_pos[:, :, 0, :], h_pos[:, :, 1, :]
        del hs, h_pos

        logits = outputs.logits[
            torch.arange(B, device=device),
            torch.as_tensor(prompt_lens, device=device) - 1, :,
        ].float()  # (B, V)
        del outputs

        raw = {
            "gate_pre": cap.stack("gate_pre"),  # (B, L, 2, d_ff)
            "sel": cap.stack("sel"),
            "down": cap.stack("down"),           # (B, L, 2, d)
        }
        del cap.captured

        if save_raw:
            np.savez_compressed(
                RAW_DIR / f"{run_tag}_batch{start:05d}.npz",
                ids=np.array([q.id for q in batch]),
                h_last=h_last.numpy(),
                h_ans=h_ans.numpy(),
                gate_pre=raw["gate_pre"].numpy().astype(np.float16),
                sel=raw["sel"].numpy().astype(np.float16),
                down=raw["down"].numpy().astype(np.float16),
            )

        gate = raw["gate_pre"].float().numpy()   # (B, L, 2, d_ff)
        sel = raw["sel"].float().numpy()
        down = raw["down"].float().numpy()
        del raw
        h_all = np.stack([h_last.float().numpy(), h_ans.float().numpy()], axis=2)
        h_all = h_all.transpose(1, 0, 2, 3)  # (B, L+1, 2, d)

        # ---- summary stats, vectorized over B, L, positions ----
        g_sq = gate * gate
        s_sq = sel * sel
        g_mu = gate.mean(-1, keepdims=True)
        g_sd = gate.std(-1, keepdims=True) + EPS
        stats = {
            "ent_gate": entropy_logits_np(gate),
            "ent_gate_sq": entropy_logits_np(np.log(g_sq + EPS)),
            "active_frac": (gate > 0).mean(-1),
            "top5_gate": topk_mass_np(g_sq, 5, axis=-1),
            "gini_gate": gini_np(gate, axis=-1),
            "max_gate": gate.max(-1),
            "kurt_gate": (((gate - g_mu) ** 4).mean(-1)) / (g_sd[..., 0] ** 4),
            "ent_sel_sq": entropy_logits_np(np.log(s_sq + EPS)),
            "l2_sel": np.linalg.norm(sel, axis=-1),
            "l2_down": np.linalg.norm(down, axis=-1),
            "cos_down_h": np.sum(down * h_all[:, 1:], axis=-1)
            / (np.linalg.norm(down, axis=-1) * (np.linalg.norm(h_all[:, 1:], axis=-1) + EPS)),
        }
        stats["down_norm_ratio"] = stats["l2_down"] / (np.linalg.norm(h_all[:, 1:], axis=-1) + EPS)
        del g_sq, s_sq, h_all

        # ---- candidate set for answer-direction alignment ----
        top = torch.topk(logits, k=min(N_COMPETITORS, V), dim=-1).indices  # (B, K)
        gold_ids = torch.tensor([a[0] for a in ans_ids], dtype=torch.long, device=device)
        cand_list = []
        for i in range(B):
            cand = list(dict.fromkeys(top[i].tolist() + [int(gold_ids[i])]))[: N_COMPETITORS + 1]
            gold_pos = cand.index(int(gold_ids[i]))
            pad = [cand[0]] * (N_COMPETITORS + 1 - len(cand))
            valid = [True] * len(cand) + [False] * len(pad)
            cand_list.append((cand + pad, gold_pos, valid))
        cand_idx = torch.tensor([c for c, _, _ in cand_list], device=device)  # (B, C)
        gold_pos = torch.tensor([g for _, g, _ in cand_list], device=device)
        valid = torch.tensor([v for _, _, v in cand_list], device=device)  # (B, C)
        D_cand = D_gpu[cand_idx]  # (B, C, d)
        D_gold = D_gpu[gold_ids]  # (B, d)
        D_gold_norm = D_gold.norm(dim=-1) + EPS

        align_stats = {k: np.zeros((B, L, 2), dtype=np.float32) for k in
                       ("cos_sel_agold", "proj_gold_norm", "mlp_margin", "ent_cand", "logit_mlp_gold")}
        ar = torch.arange(B, device=device)
        for l in range(L):
            w2 = W2_gpu[l]                       # (d, d_ff)
            a_cand = D_cand @ w2                 # (B, C, d_ff)
            a_gold = D_gold @ w2                 # (B, d_ff)
            a_gold_norm = a_gold.norm(dim=-1) + EPS
            for p in range(2):
                s = torch.from_numpy(sel[:, l, p, :]).to(device, torch.float32)   # (B, d_ff)
                s_norm = s.norm(dim=-1) + EPS
                proj = (a_gold * s).sum(-1)                     # (B,)
                cos_sel = proj / (a_gold_norm * s_norm)
                align_stats["cos_sel_agold"][:, l, p] = cos_sel.cpu().numpy()
                dn = torch.from_numpy(down[:, l, p, :]).to(device, torch.float32)
                align_stats["proj_gold_norm"][:, l, p] = (
                    proj / (dn.norm(dim=-1) * D_gold_norm + EPS)
                ).cpu().numpy()
                align_stats["logit_mlp_gold"][:, l, p] = (dn * D_gold).sum(-1).cpu().numpy()
                cand_logits = torch.einsum("bcf,bf->bc", a_cand, s)  # (B, C)
                cand_logits = cand_logits.masked_fill(~valid, -1e9)
                gold_logit = cand_logits[ar, gold_pos]
                other = cand_logits.masked_fill(
                    torch.nn.functional.one_hot(gold_pos, cand_logits.shape[1]).bool(), -1e9
                )
                align_stats["mlp_margin"][:, l, p] = (gold_logit - other.max(-1).values).cpu().numpy()
                pv = torch.softmax(
                    torch.where(valid, cand_logits, torch.full_like(cand_logits, -1e9)), -1
                )
                ent = -(pv * torch.log(pv + EPS)).sum(-1)
                align_stats["ent_cand"][:, l, p] = ent.cpu().numpy()
            del a_cand, a_gold
        for k, v in align_stats.items():
            stats[k] = v

        # ---- P features ----
        probs = torch.softmax(logits, dim=-1)
        top2 = torch.topk(probs, k=2).values
        p_gold = probs[ar, gold_ids]
        p_features = {
            "p_gold": p_gold.cpu().numpy(),
            "p_entropy": (-(probs * torch.log(probs + EPS)).sum(-1)).cpu().numpy(),
            "p_top1": top2[:, 0].cpu().numpy(),
            "p_margin": (top2[:, 0] - top2[:, 1]).cpu().numpy(),
            "p_gold_rank": ((probs > p_gold[:, None]).sum(-1) + 1).cpu().numpy(),
        }
        del logits, probs, D_cand

        # ---- assemble rows ----
        new_rows = []
        for i, q in enumerate(batch):
            row = {
                "id": q.id,
                "subj": q.subj,
                "subj_id": q.subj_id,
                "prop": q.prop,
                "obj": q.obj,
                "s_pop": q.s_pop,
                "o_pop": q.o_pop,
                "n_aliases": len(q.aliases),
                "prompt_tokens": prompt_lens[i],
                "subject_tokens": len(tokenizer(q.subj, add_special_tokens=False)["input_ids"]),
            }
            for k, v in p_features.items():
                row[k] = float(v[i])
            for stat_name, arr in stats.items():
                for p, pname in enumerate(POSITIONS):
                    for l in range(L):
                        row[f"m_{pname}_l{l}_{stat_name}"] = float(arr[i, l, p])
            rows.append(row)
            new_rows.append(row)
        fout.write("\n".join(json.dumps(r) for r in new_rows) + "\n")
        fout.flush()

        del gate, sel, down, stats, new_rows
        torch.cuda.empty_cache()
        done = min(start + batch_questions, n)
        if (start // batch_questions) % 5 == 0 or done == n:
            print(f"[feat] {done}/{n} questions", flush=True)

    fout.close()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-questions", type=int, default=8)
    ap.add_argument("--model", type=str, default=MODEL_NAME)
    ap.add_argument("--tag", type=str, default="pilot")
    ap.add_argument("--skip-gen", action="store_true")
    ap.add_argument("--skip-feat", action="store_true")
    ap.add_argument("--fewshot", action="store_true", help="few-shot short-answer prompt")
    ap.add_argument("--resume-feat", action="store_true", help="resume feature extraction")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")
    if device == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    questions = load_questions()
    if args.limit:
        questions = questions[: args.limit]
    print(f"loaded {len(questions)} questions")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="sdpa"
    ).to(device)
    model.eval()
    build_prompts(tokenizer, questions, fewshot=args.fewshot)
    print(
        f"model: {args.model} ({model.config.num_hidden_layers} layers, "
        f"d_ff={model.config.intermediate_size})"
    )

    if not args.skip_gen:
        frac, greedy, samples = generate_correctness(
            model, tokenizer, questions, args.k, args.batch_questions, device
        )
        conf = generate_confidence(model, tokenizer, questions, args.batch_questions, device)
        import pandas as pd

        labels = pd.DataFrame({
            "id": [q.id for q in questions],
            "subj_id": [q.subj_id for q in questions],
            "prop": [q.prop for q in questions],
            "frac_correct": frac,
            "greedy_correct": greedy,
            "confidence": conf,
        })
        labels.to_csv(RESULTS_DIR / f"{args.tag}_labels.csv", index=False)
        with open(RESULTS_DIR / f"{args.tag}_samples.jsonl", "w") as f:
            for q, ss in zip(questions, samples):
                f.write(json.dumps({"id": q.id, "question": q.question, "samples": ss}) + "\n")
        print(f"mean frac_correct={frac.mean():.3f}; greedy acc={greedy.mean():.3f}; "
              f"mean confidence={np.clip(conf, 0, 100).mean():.1f}")

    if not args.skip_feat:
        import pandas as pd

        rows = extract_features(
            model, tokenizer, questions, args.batch_questions, device, args.tag,
            resume=args.resume_feat,
        )
        feats = pd.DataFrame(rows)
        feats.to_parquet(RESULTS_DIR / f"{args.tag}_features.parquet", index=False)
        print(f"wrote features: {feats.shape}")


if __name__ == "__main__":
    main()
