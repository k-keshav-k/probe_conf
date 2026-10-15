# Can better retrieval make confidence more useful?

**Selective MLP retrieval: toy experiments E1-E3**

Keshav Krishna · New York University · 15 October 2026

**Main result.** Training improved some retrieval scores and brought confidence closer to the true success probability, but the report added very little to a predictor using answer probabilities. The added-information claim is not supported in this toy setting.

## 1. Question and setup

Can a model learn to retrieve the relevant fact **and recognize when it will succeed**? E1 tests retrieval on new fact assignments; E2 tests confidence about the next sampled answer; E3 tests confidence about an answer after a specified future memory change.

The artificial world has 512 subjects, four attributes and 128 answer symbols (384/64/64 train/development/test subjects). During training, 30% of bindings share one answer across all attributes, creating an irrelevant-information shortcut. Fresh assignments break that shortcut. Controlled changes weaken or swap asked or unasked attributes.

The model has one decoder block (residual width 32; MLP width 256). Rank-8 adapters change only the MLP gate/up projections after a common supervised warm start. A separate two-layer reporting MLP (width 256) outputs confidence; **this is not native verbal confidence in a pretrained LLM**. E1-E3 average five training seeds; the probe diagnostic uses one.

## 2. Training arms and objectives

All arms use the same reporting procedure. C versus B tests the combined method; C versus E isolates the added consistency penalty.

| Arm | Retrieval training |
| --- | --- |
| A | Freeze retrieval; train only the reporter. |
| B | Optimize correctness reward with a reference-policy penalty. |
| C | B plus labeled examples with irrelevant facts changed, and a consistency penalty. |
| D | Supervised correct-answer and consistency losses; no B reward. |
| E | B plus C's labeled examples, without the consistency penalty. |

Let $p$ and $p'$ be answer distributions before and after changing unasked attributes; $p_{\rm ref}$ is the reference policy. Let $Y$ indicate correctness and $\mathrm{CE}(p)=-\log p(a_{\rm gold})$. The objectives are:

$$
\begin{aligned}
J_B &= \mathbb{E}[Y]-0.1\,D_{\rm KL}(p\parallel p_{\rm ref}),\qquad L_B=-J_B,\\
L_{\rm sel} &= \mathrm{CE}(p)+\mathrm{CE}(p')+D_{\rm KL}(\operatorname{sg}[p]\parallel p'),\\
L_C &= L_B+5L_{\rm sel},\qquad L_D=L_{\rm sel},\qquad
L_E=L_B+5[\mathrm{CE}(p)+\mathrm{CE}(p')].
\end{aligned}
$$

The consistency term discourages changes caused by irrelevant facts; $\operatorname{sg}$ stops the gradient through its clean branch only. D also differs in weighting and regularization, so it is not a perfectly isolated RL-versus-supervision comparison. The small answer set permits exact $\mathbb{E}[Y]=p(a_{\rm gold})$: retrieval uses an expected-reward gradient, not sampled GRPO.

The confidence reward is $R(c,Y)=-(c-Y)^2$. In practice, the reporter minimizes $L_{\rm report}=(c-\bar Y_8)^2$ against eight independent outcomes from a frozen answer policy. Confidence $c$ is the mean over 21 output values (0 to 1, step 0.05). Reporting gradients update only the report head; retrieval gradients update only the adapters.

**Settings:** 1,000 warm-start steps; three rounds of 1,200 retrieval and 3,000 reporting steps; sampling temperature 1; 16 evaluation draws per item.

## 3. E1: retrieval improves, but attribution is limited

On fresh assignments, C beats B by **2.6 percentage points in greedy accuracy** and **1.6 points in following valid fact swaps**, positive in all five seeds. Irrelevant-swap stability is unchanged. However, E nearly matches C in E2, so the consistency term's separate retrieval benefit is not established. The original selectivity score is confounded because weakening unasked attributes also removes B's shortcut.

<!-- pagebreak -->

## 4. E2: little added value from confidence

The reporter predicts whether the next **sampled** answer will be correct. The ordinary baseline uses relation identity, maximum answer probability, entropy and top-two probability margin. It receives neither the gold answer nor the intervention label; a privileged baseline is separate. Coherent false memories are excluded from the main endpoint.

**Reading the table.** Greedy accuracy scores the most likely answer; sampled success scores a draw from the answer distribution. Brier is squared error against correct/incorrect outcomes (lower is better; always saying 50% gives 0.250). **Added value** is baseline Brier minus Brier after adding confidence (higher is better). **Probability error** measures squared distance from true success probability, not calibration alone. Answer scores are clean; reporting scores cover the main conditions.

| Arm | Greedy | Sampled | Report Brier | Baseline Brier | Added value | Probability error |
| --- | --- | --- | --- | --- | --- | --- |
| A | 58.8% | 40.8% | 0.267 | 0.181 | +0.00049 | 0.139 |
| B | 61.7% | 43.9% | 0.276 | 0.182 | +0.00033 | 0.147 |
| C | 64.1% | 42.1% | 0.268 | 0.182 | +0.00038 | 0.128 |
| D | 63.8% | 41.9% | 0.269 | 0.181 | +0.00034 | 0.128 |
| E | 64.0% | 43.8% | 0.274 | 0.182 | +0.00049 | 0.139 |

C's probability error improves over B by 0.019 and E by 0.011, in all five seeds. Yet its added value is only 0.00038, below E's. Every raw report is worse than always saying 50%; recalibration improves Brier to 0.225-0.230, still far behind the baseline. C's greedy gain does not extend to sampled success.

Tracking is partial: target weakening lowers C's actual success by 0.377 but confidence by only 0.105 (about 70% sign agreement). On separately evaluated false memories, confidence is about 19% versus actual success of 0.7%.

## 5. E3: predicting success before a change

The reporter sees the **clean state plus the upcoming intervention's type and strength**. The baseline receives clean probability features and the same description. Neither sees the changed state. Training uses weakening strengths 0.2/0.5/0.8 and swap scales 1/2; unseen tests use 0.35/0.65 and scale 3. These are new strengths, not new intervention families.

| Arm | Report Brier | Baseline Brier | Added: seen strengths | Added: unseen strengths | Probability error |
| --- | --- | --- | --- | --- | --- |
| A | 0.243 | 0.173 | +0.00000 | +0.00010 | 0.125 |
| B | 0.252 | 0.175 | +0.00001 | +0.00008 | 0.134 |
| C | 0.245 | 0.175 | +0.00008 | +0.00040 | 0.119 |
| E | 0.251 | 0.175 | +0.00012 | +0.00038 | 0.128 |

Errors are at seen strengths on test subjects; D was not run. On unseen strengths, C exceeds B's added value by 0.00033 (5/5 seeds), but E by only 0.00002 (2/5). Extra supervision remains a sufficient explanation; **no clear added-value benefit of the consistency penalty appears**. Fresh assignments also show little gain.

## 6. Diagnostic and conclusion

In a separate **one-seed** diagnostic on arms B/C/E, hidden-state probes also trail the baseline: Brier 0.236-0.237 for a linear probe, 0.204-0.211 for a neural probe trained on exact success probabilities, and 0.210-0.215 with sampled labels, versus 0.144-0.151 for the baseline. These ranges span arms, not confidence intervals. Exact labels do not close the gap, but probe failure **does not prove information is absent**; answer probabilities are themselves computed from the hidden state.

**Conclusion:** the combined method improves some retrieval scores and probability estimation, but does not demonstrate useful reporting value beyond the tested probability baselines. Small mean gains are not equivalence proofs. Reporter learning, readout capacity and generalization remain unresolved. Scaling the same method needs a specific explanation and test of this failure, not just more compute.

**Evidence:** [E1](../runs/section33_v1/selective_metacognition/e1.json), [audited E2](../runs/section33_v1/selective_metacognition/e2.json), [E3](../runs/section33_v1/selective_metacognition/e3.json), [probe diagnostic](../runs/selective_metacognition/e3_probe_diagnostic.json), and [full proposal](SELECTIVE_MLP_METACOGNITION_PROPOSAL.md).
