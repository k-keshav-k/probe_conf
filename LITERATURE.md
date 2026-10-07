# Literature Notes — Confidence, Difficulty, and Probing Factual Recall

Compiled 2026-09-28 for the `probe_conf` pilot. Three threads:
(A) the mechanistic-recall backbone, (B) the confidence-vs-difficulty thesis,
(C) probing internal states for correctness — and what it means for the proposed
MLP-interaction probe.

Evidence review, 2026-10-03: this is a research notebook. The completed
experiment is described in `report/REPORT.md`. Novelty, causal mechanisms, and
difficulty-independent confidence are hypotheses, not pilot findings. The main
report references were verified against primary records. Detailed theorem
claims in this notebook were not independently rederived.

---

## A. Mechanistic accounts of factual recall

### A1. The reference paper: Geometric Factual Recall

**Ravfogel, Yehudai, Bruna, Bietti. "Geometric Factual Recall in Transformers."**
arXiv:2605.12426 (May 2026).

Main claims (theory + synthetic experiments on a single-layer transformer):

- For a single layer memorizing `N` subjects x `R` relations over a **shared**
  attribute set, an embedding dimension `d = O(R log N)` suffices — *logarithmic* in
  the number of facts — provided embeddings are **learnable**.
- Subject embeddings encode **linear superpositions** of their `R` attribute vectors;
  the MLP is **not** a key-value associative memory but a **relation-conditioned
  selector** implemented by ReLU/SiLU gating that extracts the queried attribute
  block.
- Gradient descent discovers exactly this structure: per-relation linear readouts
  decode attributes from subject embeddings almost perfectly; causal interventions
  show the MLP follows swaps on the queried relation while leaving others intact;
  and a freeze-and-swap experiment shows the MLP **transfers zero-shot to new
  bijections** when subject embeddings are re-initialized (generic selector, not a
  memorizer).
- Multi-hop: no-CoT solutions pay a capacity-depth tradeoff
  (`W >= max(B_global, B_local) - R*D`), and chain-of-thought provably reduces the
  multi-hop problem back to the single-hop capacity regime.

**Real-LM probing section (5.3) — closest methodology to ours:**

- Task: per-relation **linear readout** of the gold answer's first token embedding
  (`W_r s_l + b_r` vs. `D[t_o]`, rank-512 low-rank factorized affine map, cosine
  objective), probing every layer, at the **last subject token**.
- Subject-disjoint 80/20 split per `(category, relation)`.
- Findings: layer-0 is weak but non-trivial; readout quality climbs through middle
  layers and peaks at roughly 50–80% depth for people (best-layer MRR ~0.69 for
  Qwen2.5-0.5B people vs. 0.44 at layer 0). Crucially, output-vocabulary directions
  (LM-head rows) are linearly decodable from the subject state — stronger than the
  older claim in Hernandez et al. (2023) that only an intermediate state is decodable.

**Implications for us:**

1. The ROME-style "query dot W2 columns" story is **not** what this paper supports.
   The MLP is a gating selector over attribute directions already present in the
   residual stream. If we want "interaction" features that are mechanistically
   grounded, the right objects are the **gate pre-activations** `W1 x` (key-match
   coefficients), the **gates** `silu(W1 x)`, the **gated value writes**
   `silu(W1 x) * (W3 x)`, and their projection onto answer directions.
2. Their readout uses the **subject token**; recall-relevant information is enriched
   through middepth layers there. We should probe at both the subject-last token and
   the last prompt token.
3. Subject-disjoint splits and per-relation breakdowns are the established protocol;
   adopt them so results are comparable.

### A2. Key-value-memory and ROME lineage

- **Geva et al. "Transformer Feed-Forward Layers Are Key-Value Memories."**
  EMNLP 2021, arXiv:2012.14913. FFN layers act as key-value memories: the rows of
  `W1` are keys matched by the residual stream, the corresponding rows of `W2` are
  value distributions over the output vocabulary. This is the origin of the
  "query x key dot product" picture.
- **Meng et al. "Locating and Editing Factual Associations in GPT" (ROME).**
  NeurIPS 2022, arXiv:2202.05262. Causal tracing localizes subject-attribute recall
  to mid-layer MLPs; knowledge is injected by a rank-one edit of `W2` at the
  "critical" layer. Establishes that MLP second matrices can act as an editable
  associative store for facts.
- **Geva et al. "Dissecting Recall of Factual Associations in Auto-Regressive LMs."**
  EMNLP 2023, arXiv:2304.14767. Three-step view: attribute extraction, subject
  enrichment (attention moves the subject's representation to the final token), and
  attribute transfer to the prediction. Supports probing mid-late layers and the
  subject token.
- **Nichani, Lee, Bietti. "Understanding Factual Recall via Associative Memories."**
  ICLR 2025, arXiv:2412.06538. With **fixed random** embeddings, transformer
  associative memories need parameters linear in the number of facts; complements
  the geometric paper, whose point is that *learned* embeddings break the linear
  parameter scaling.

**Tension to keep in mind.** Two explanations of the same machinery coexist:
ROME/associative-memory (facts stored in MLP weights) vs. geometric-superposition
(facts in embedding geometry, MLP as a cheap selector). The pilot's "MLP interaction
features" are justified under either story, but the *specific features* differ:
key-match coefficients + gate sparsity under the geometric story, value-column
alignment under the ROME story. Our feature set covers both.

---

## B. Confidence vs. difficulty

### B1. Core references already in the notes

- **Kadavath et al. "Language Models (Mostly) Know What They Know."**
  arXiv:2207.05221 (2022). Models' P(True) self-evaluations are informative but
  imperfect; calibration degrades out of distribution.
- **Tian et al. "Just Ask for Calibration."** EMNLP 2023, arXiv:2305.14975.
  Eliciting verbalized probabilities gives better calibration than sequence
  likelihood, especially for RLHF models, but the verbal channel is format-sensitive.
- **Xiong et al. "Can LLMs Express Their Uncertainty?"** ICLR 2024, arXiv:2306.13063.
  LLMs are systematically overconfident when verbalizing confidence.
- **RLMF: "Reinforcement Learning with Metacognitive Feedback Elicits Faithful
  Uncertainty Expression in LLMs."** arXiv:2606.32032 (2026). RL on the *quality of
  the model's self-assessment* improves agreement between expressed and intrinsic
  confidence. This is the "RLMF-style setup" referenced in the meeting notes.

### B2. The "confidence is a difficulty proxy" claim

- **"Reported Confidence in LLMs Tracks Commitment More Than Correctness."**
  arXiv:2606.29490 (2026). Recent direct evidence that verbalized confidence is
  better explained by response commitment/stability than by correctness — close to
  the central thesis of the PCA paper; worth reading closely before the next meeting.
- **"Causal evidence that language models use confidence to drive behaviour."**
  Nature Machine Intelligence (2026), doi:10.1038/s42256-026-01293-x. Contains
  analyses showing the explanatory power of verbal confidence is largely driven by
  question difficulty; also shows confidence causally drives abstention.
- **Mielke et al. "Reducing Conversational Agents' Overconfidence Through Linguistic
  Calibration."** TACL 2022, arXiv:2204.05862. Hidden states *do* encode a
  correctness signal, and verbalized confidence can be improved by calibrating the
  wording against internal states — useful counterpoint: the signal exists; the
  verbal channel is lossy.
- **Kuhn et al. "Semantic Uncertainty."** ICLR 2023, arXiv:2302.09664; and
  **Kossen et al. "Semantic Entropy Probes."** arXiv:2406.15927 (2024). Consistency
  across sampled generations approximates semantic uncertainty; SEPs recover it
  from hidden states, but only for short generations, and the probe can be
  brittle.
- **Calibration saturation.** Verbalized confidence collapses onto a few coarse
  values (e.g., 0.9/1.0); this alone can make it look "difficulty-tracking" because
  easy facts dominate the high-confidence mass (Wang et al., 2025, "Calibrating
  Verbalized Confidence with Self-Generated Distractors"; see also DINCO).

### B3. Where the pilot fits

The meeting notes' experimental logic (bins, surface predictor, residual
diagnostic) is a clean way to test "is verbalized confidence more than a popularity
proxy?" The probe pilot asks the *complementary* question: can a *trained* readout
of internal computation predict correctness beyond difficulty, and does the
interaction geometry add anything over a plain hidden-state readout?

---

## C. Probing internal states for correctness / hallucination

- **Azaria & Mitchell. "The Internal State of an LLM Knows When It's Lying."**
  EMNLP Findings 2023, arXiv:2304.13734. First prominent truthfulness probe on
  hidden activations. Establishes the "internal states know" baseline.
- **Chen et al. "INSIDE: LLMs' Internal States Retain the Power of Hallucination
  Detection."** ICLR 2024, arXiv:2402.03744. EigenScore: eigenvalues of the
  covariance of hidden states across sampled answers; a self-consistency signal
  computed from internal geometry. Adjacent to "distribution of internal activity"
  features, though on hidden states rather than MLP gates.
- **Orgad et al. "LLMs Know More Than They Show: On the Intrinsic Representation of
  LLM Hallucinations."** ICLR 2025, arXiv:2410.02707. Truthfulness signal is often
  present in internal states but not linearly decodable at the answer token;
  mismatch between internal and external behavior.
- **Zhang et al. "ICR Probe."** ACL 2025, arXiv:2507.16488. Quantifies each module's
  *contribution* to the residual-stream update (attention vs. FFN) and probes the
  cross-layer vector of contributions. Closest existing work to "probing module
  interaction signals": it uses module *update norms/alignment*, not the MLP gate
  distribution itself.
- **CLAP: Cross-Layer Attention Probing.** arXiv:2509.09700 (2025). Treats the
  residual stream across layers as a token sequence and attention-pools it; improves
  hallucination detection and fine-grained detection among sampled responses.
  Strong evidence that **cross-layer aggregation** of activations is useful, which
  supports the architecture direction in the meeting notes.
- **ACT-ViT.** NeurIPS 2025. Pools the whole layer x token x dim activation tensor
  and runs a ViT; demonstrates cross-layer/token processing and some zero-shot
  transfer to unseen LLMs.
- **"Automatic Layer Selection for Hallucination Detection."** arXiv:2605.26366
  (2026). Layer choice is dataset-dependent; FEPoID picks the first intrinsic-
  dimension peak. Useful for avoiding cherry-picked "best layer" claims.
- **"Do Androids Know They're Only Dreaming of Electric Sheep?"** ACL Findings 2024.
  Probes at every layer and for every module type; FFN probes are slightly stronger
  than attention out-projection probes for grounded hallucination. Also warns that
  probes are domain-narrow.

### C1. Novelty audit for "probing MLP interaction features"

Candidate distinctions that need a systematic prior-art check:

- Probing the **distribution/geometry of MLP gate activations** (key-match
  coefficients `W1 x`, gates `silu(W1 x) * (W3 x)`, sparsity/entropy/competition
  among MLP features) to predict correctness or confidence.
- Using **answer-direction projections of the MLP write** (alignment of the MLP's
  residual contribution with the gold-token unembedding vs. competitor tokens) as a
  confidence feature.

Adjacent prior art that must be cited/positioned against:

- Hidden-state probes (Azaria & Mitchell; INSIDE; SEPs; CLAP; ACT-ViT; ICR).
- Module-contribution probes (ICR) — different quantity, same motivation.
- Static probes at one layer/token (most of the above); cross-layer attention over
  *hidden states* (CLAP) is the nearest architectural neighbor to the proposed
  auxiliary model.

**Theoretical caveat.** `gate_pre = W_gate x` is linear in normalized MLP
input `x`. Full column rank would give unrestricted linear classifiers on these
vectors the same function class. Rank was not measured. This does not imply
equal test AUROC with finite data and L2 regularization. The saved hidden probe
reads a post-block state, not `x`. PCA changes the raw comparison further.
Possible design differences include:

1. the **nonlinear** gating structure (`silu`, products with `W3 h`),
2. **concentration/competition statistics** (entropy, sparsity, top-k mass) —
   nonlinear summaries of the interaction,
3. **cross-layer aggregation** of the interaction pattern,
4. **alignment with answer directions** (geometric relation to the readout).

This is a useful refinement of the meeting-notes plan: the pilot is really testing
whether *nonlinear summaries of MLP selection geometry* carry correctness signal.

---

## D. Datasets

- **PopQA** (Mallen et al., ACL 2023, arXiv:2212.10511; HF `akariasai/PopQA`).
  14k questions over 16 Wikidata relations; includes `s_pop` (Wikipedia monthly
  pageviews), aliases, and gold answer sets. **Caveat:** PopQA's 16 relations do
  **not** include *date of birth* (they are: occupation, place of birth, genre,
  father, country, producer, director, capital of, screenwriter, composer, color,
  religion, sport, author, mother, capital). The pilot must either substitute
  (e.g., *place of birth* / *religion*) or extend PopQA from Wikidata for
  *date of birth*. Recommended pilot set of person relations:
  **father, mother, place of birth, occupation**. Relation filtering alone does
  not verify real-person status. Mythological subjects occur in the saved sample.
- **EntityQuestions** (Sciavolino et al.) — a long-tail alternative/complement.

---

## E. One-paragraph summary for the group

The geometric-recall paper reframes what a fact "is" inside a transformer — learned
embeddings carry a linear superposition of attributes and the MLP is a cheap
relation-conditioned gate — which means the natural object for a confidence probe is
the **geometry of the gate distribution** (competition among matched keys, alignment
of the gated write with the answer direction) rather than a static hidden state.
Existing internal-state and module-contribution work motivates this comparison.
The notebook does not establish the absence of prior MLP-distribution probes or
difficulty controls. The pilot gives an offline comparison of selected features.
It does not distinguish memory theories or establish live-use confidence.
