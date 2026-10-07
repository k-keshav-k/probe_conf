# Model Confidence and Difficulty in Factual QA with Probe Design

Historical meeting notes. The statements below record discussion and hypotheses.
The completed pilot and its evidence limits are in `report/REPORT.md`.
The original transcript is in `meet.original.txt`.

## 1. Research Context: Confidence vs. Difficulty

**Central thesis.** LLM confidence reports predominantly track objective question
difficulty, not genuine internal confidence.

- **Multi-model PCA paper** (not yet shared with the group): collected confidence
  reports from ~30 models on PopQA-style data.
  - Formed a models x confidence-report matrix and ran PCA; one principal component
    captures most of the variance.
  - That single component correlates strongly with difficulty measures, not with
    model-specific internal states.
- **Consensus in recent literature:** confidence is largely a difficulty proxy, not a
  faithful self-assessment.

**Current experimental scope**

- PopQA dataset, restricted to person subjects.
  - Rationale for the restriction: fame and popularity are tractable proxies for
    difficulty.
- 4 relations used: father, mother, date of birth, and one additional
  popularity-linked relation.
- Approach is similar in spirit to RLMF-style setups using structured knowledge
  probing.

## 2. Measuring Confidence vs. Correctness

Two independent scores are computed per question:

| Score | How it is computed |
| --- | --- |
| **Fraction correct** | 30 generations per prompt; fraction of correct answers used as a consistency/knowledge score. |
| **Confidence report** | Model prompted separately and directly to give a confidence score from 1 to 100. |

- The two scores are kept independent to avoid conflating recall with self-reported
  confidence.

**Surface-level predictor** — trained via logistic regression on observable features
only:

- Wikipedia monthly page views
- Number of languages the article exists in
- Name collision count
- Relation type
- Answer set size
- Prompt length

The predictor outputs a predicted fraction-correct score based purely on these surface
signals.

**Difficulty bins**

- Constructed from popularity features alone.
- Within each bin, popularity bias is held constant across all examples.
- Both the confidence score and the surface predictor share the same popularity bias
  inside a bin, so that bias cancels when comparing them.
- **AUROC comparison:** if confidence AUROC exceeds surface predictor AUROC within
  bins, confidence carries internal signal beyond what surface features explain.
- **Residual diagnostic:** subtract the surface-level prediction from the observed
  fraction correct, then correlate the residual with the confidence score.
  - Tests whether confidence tracks something beyond what the logistic regression
    already captures.
  - This is an evaluation step, not a machine learning training step.

## 3. Novel Direction: Probing MLP Interactions

### 3.1 Core problem with standard probing under difficulty control

- When probes are trained within a single difficulty bin, difficulty is constant
  across all examples in that bin.
- A probe cannot use difficulty as a predictive feature if it does not vary, so it
  loses its strongest signal.
- Analogy: predicting height from weight works in general, but if all subjects weigh
  exactly 70 kg, weight is useless as a feature.
- Result: probes conditioned on difficulty bins perform worse — not because confidence
  is absent, but because the probe has no remaining features to rely on.
- **Fundamental conundrum:** difficulty is the best predictor of correctness, but
  conditioning on it removes the feature entirely.

### 3.2 Why standard confidence prompting may be insufficient

- When the model is asked to report confidence as part of the same forward pass as the
  factual question, the confidence token is generated using the same internal
  mechanisms as the answer.
- The model has no independent component that observes its own retrieval process; it
  only sees the output of that process, not the interactions that produced it.
- The model is, by construction, blind to its own key-value interactions in the MLP
  layers.

### 3.3 Proposed direction: an auxiliary probe over MLP interaction features

- Train a small auxiliary model to probe MLP interaction features directly.
- Grounded in the mechanistic interpretability view of factual recall (ROME paper): a
  query vector interacts with columns of the MLP weight matrix W2 via dot products.
  - High-confidence retrieval expected when one column dominates (high dot product,
    others near zero).
  - Uncertain retrieval expected when multiple columns have roughly equal dot-product
    magnitudes.
- A probe trained directly on this dot-product distribution, rather than on
  residual-stream hidden states, could in principle report confidence that is
  independent of difficulty.
- The auxiliary model does not need to understand how knowledge is distributed; it
  learns the pattern via gradient descent across all MLP layers and all factual recall
  instances.

### 3.4 Architecture open questions

- Could be a simple linear probe applied to MLP interaction features across layers.
- Could be a nonlinear pooling mechanism, e.g. an attention layer over intermediate MLP
  outputs that aggregates across all layers before producing a confidence estimate.
- Knowledge is distributed across layers with no single localized position, so
  cross-layer aggregation is likely necessary.
- Whether the MLP selection and retrieval mechanism is fully understood remains an open
  question in the field.

### 3.5 Key test

- Does the resulting confidence report become less sensitive to objective difficulty
  than a standard prompted confidence score?
- Training data for the auxiliary probe should control for difficulty to prevent the
  probe from falling back on difficulty features.
- Parallels the difficulty-binning logic from the main experiment.

### 3.6 Relevant literature

- **ROME paper:** established the factual recall mechanism and used it to inject new
  knowledge by editing MLP weights.
- **Mechanistic interpretability work more broadly:** motivates the idea that MLP
  columns act as knowledge stores that can be read by a query vector.
- No known prior work probes the interaction features between query and MLP columns
  directly, as opposed to probing hidden states or residual-stream activations.

## 4. Preliminary Experiment and Next Steps

Proposed preliminary experiment before committing to the full direction:

- Train standard linear probes on hidden states across all layers as a baseline,
  predicting whether the model will answer correctly.
- Train probes on MLP interaction features instead, e.g. dot-product distributions
  between query and W2 columns, entropy of those distributions, or similar summary
  statistics.
- Compare AUROC of both probe types without yet controlling for difficulty.
  - If the MLP interaction probe performs comparably to the hidden-state baseline, the
    signal is present and the direction is worth pursuing further.
  - If performance drops substantially, the direction is likely not viable and should
    be abandoned early.
- Difficulty control can be layered in as a second step once basic signal is confirmed.

## 5. Broader Open Question

- Whether analogous confidence calibration problems exist for LLM agents, where the
  model may be uncertain not just about a factual answer but about multi-step reasoning
  or action selection.
- No literature reviewed on this yet; flagged as a parallel research thread worth
  exploring.

## 6. Overall Assessment

- Direction is exploratory, open-ended, and carries meaningful risk of not working.
- Recommended to take a few more days to think independently, note related ideas, and
  reconvene before committing further effort.

## Next Steps

1. **Run the preliminary probe experiment** comparing hidden-state vs. MLP interaction
   features:
   - Train standard linear probes on hidden states as the baseline, then train probes
     on MLP dot-product distributions or entropy.
   - Compare AUROC to establish whether the signal exists before pursuing the full
     architecture.
2. **Take more time to evaluate the MLP probing direction** before committing:
   - The direction is open-ended and uncertain; note down related ideas and share them
     with the group before deciding whether to pursue it as an independent study
     project.
3. **Search literature on confidence calibration for LLM agents:**
   - Parallel questions may exist beyond single-model factual confidence; share any
     relevant findings with the group.
