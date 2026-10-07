# Agent Confidence & Calibration — Literature Scan (2024–2026)

Compiled 2026-10-03 for the `probe_conf` agent-confidence thread (meeting notes §5).
Scope: confidence/calibration for LLM agents; process supervision and step-level
correctness probes; uncertainty propagation across multi-step reasoning; action-level
confidence and tool-use error detection; internal-state probing for agent trajectories.
Excluded by request: factual-recall probing, SAEs, gradients, probe methodology.

**Verification legend** (applies to every row):

- **[V]** verified this session on the arXiv abs page, ACL Anthology, or publisher page.
- **[P]** title + ID seen on an arXiv HTML/PDF or alphaXiv page inside fetched search
  results; consistent, but the abs page was not opened.
- **[U]** secondary source only (blog, citation list, search snippet). Verify before citing.

AlphaXiv pages/overviews were used for: 2601.15778, 2602.05073, 2609.09448, 2609.07395,
2604.05952, 2404.09127, 2506.17419, 2412.01033, 2504.05419.

---

## 1. Agent confidence and calibration

| Paper | arXiv / venue | Year | One-sentence finding |
|---|---|---|---|
| Uncertainty Quantification in LLM Agents: Foundations, Emerging Challenges, and Opportunities (Oh et al.) | 2602.05073 [V] | 2026 | First general formulation of agent UQ; identifies four agent-specific challenges (estimator choice, heterogeneous entities, interactive uncertainty dynamics, missing fine-grained benchmarks) and shows naive averaging of per-step uncertainty is close to random on τ²-bench. |
| Uncertainty Quantification for LLM Agents: A Taxonomy, an Evaluation Protocol, and an Empirical Study (Lin et al.) | 2609.07395 [V] | 2026 | Introduces TC-ECE and a counterexample that step-level calibration does not imply trajectory-level calibration; agent self-reported confidence does not consistently beat a simple baseline. |
| Agentic Confidence Calibration (Zhang et al.) | 2601.15778 [V] | 2026 | Defines agentic confidence calibration and proposes HTC, a simple model over trajectory process features (entropy dynamics, confidence gradients, stability) that beats final-step confidence across eight benchmarks. |
| Agentic Uncertainty Quantification (Zhang et al.) | 2601.15703 [V] | 2026 | Dual-process AUQ turns verbalized uncertainty into active control: memory propagates confidence across steps, and reflection is only triggered when uncertainty is high. |
| The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents (Xuan et al.) | 2601.07264 [V] | 2026 | Evidence tools (e.g., search) systematically induce overconfidence while verification tools (e.g., code) reduce miscalibration; RL fine-tuning jointly on accuracy and calibration generalizes across tool types. |
| Do Agents Know When They Succeed? Calibrating Agent Confidence from Internal Representations | 2609.09448 [V] | 2026 | LTD (residual-stream dynamics across a trajectory) and ARP (probe at action decisions) beat log-prob and sequence-confidence baselines on Bash/SQL/Python with Qwen2.5-Coder-7B/14B and DeepSeek-Coder. |
| Confidence Calibration and Rationalization for LLMs via Multi-Agent Deliberation | 2404.09127 [P] | 2024 | Post-hoc, training-free Collaborative Calibration uses tool-augmented expert debate and rationalized confidence adjustment to improve both accuracy and calibration. |
| Towards Uncertainty-Aware Language Agent (UALA) (Han et al.) | 2401.14016 [V] | 2024 | Uses UQ over Think–Act–Observe to decide whether to accept an answer or keep querying tools, improving accuracy while cutting tool calls; explicitly reports that verbalized confidence is an unreliable proxy. |
| Uncertainty of Thoughts (Hu et al.) | 2402.03271 [V] | 2024 | Models uncertainty over intermediate states to drive active information seeking in multi-step planning (NeurIPS 2024). |
| Uncertainty Calibration for Tool-Using Language Agents (Liu et al.) — ProbeCal | Findings of EMNLP 2024 ([ACL](https://aclanthology.org/2024.findings-emnlp.978/)) [V] | 2024 | Identifies prompt design and execution-trace selection as miscalibration sources and calibrates tool-use success estimates; **no arXiv ID located** (don't cite as arXiv). |
| Position: Uncertainty Quantification Needs Reassessment for LLM Agents (Kirchhof et al.) | ICML 2025 ([PMLR](https://proceedings.mlr.press/v267/kirchhof25b.html)) [V] | 2025 | Argues the aleatoric/epistemic framing is insufficient for interactive agents and calls for uncertainty about context, interaction, and communication; **no arXiv ID located**. |
| Towards Trustworthy Report Generation: A Deep Research Agent with Progressive Confidence Estimation and Calibration | 2604.05952 [P] | 2026 | Integrates per-step confidence updates into a THINK→SEARCH→READ loop, lowering N-ECE on xBench-DeepSearch relative to GPT-4o/Qwen-VL baselines. |

**Takeaway:** a 2026 literature has formed, but it mostly scores *verbalized* or
*log-prob-derived* confidence from outside; internal-state agent confidence is only a
handful of papers (2609.09448, 2605.09252, 2601.05214, 2607.06503), and two of those use
Qwen2.5 — the same family as this project.

---

## 2. Process supervision and step-level correctness probes

| Paper | arXiv / venue | Year | One-sentence finding |
|---|---|---|---|
| ReProbe: Efficient Test-Time Scaling of Multi-Step Reasoning by Probing Internal States of LLMs | 2511.06209 [V] | 2025 (v5 2026; ACL 2026) | A <10M-param transformer probe on frozen hidden states estimates step credibility across math, planning, and QA, matching PRMs up to 810× larger. |
| Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification | 2504.05419 [V] | 2025 | Hidden states at intermediate answer positions encode answer correctness (linearly), with calibrated scores and "look-ahead" to before the answer is formulated; used for a 24% token reduction. |
| Process Supervision of Confidence Margin for Calibrated LLM Reasoning (RLCM) | 2604.23333 [V] | 2026 | RL with a margin-based process reward over intermediate-budget completions (via a lightweight attached confidence probe) improves calibration without sacrificing accuracy; process supervision beats final-step-only supervision. |
| Know What You Don't Know: Uncertainty Calibration of Process Reward Models | 2506.09338 [P] | 2025 | Shows state-of-the-art PRMs systematically overestimate step success and fixes it with quantile regression, enabling instance-adaptive compute allocation. |
| Distributional Process Reward Models: Calibrated Prediction of Future Rewards via Conditional Optimal Transport | 2605.06785 [P] | 2026 | Learns a monotone conditional quantile function over PRM success probabilities, giving crossing-free confidence bounds at arbitrary levels. |
| Process Rewards with Learned Reliability (BetaPRM) | 2605.15529 [P] | 2026 | A distributional PRM outputs both step success probability and a concentration/reliability parameter, enabling reliability-aware compute allocation. |
| Inducing Process Supervision from Outcome-Only Reinforcement Learning (TIPS) | 2609.36641 [V] | 2026 | Outcome-only RL on generated step-level labels yields a generative PRM reaching 85.2 F1 on ProcessBench with 3.2K trajectories, including agent benchmarks. |
| RAWR: Reward Assignment Without Rollouts in Verifiable Domains (MCNIG) | 2603.17815 [P] | 2026 | Automatic step-quality labeling without rollouts cuts process-supervision cost to O(N) at comparable label quality. |
| The Hidden Bias of Process Reward Models: PRISM | 2606.09078 [P] | 2026 | Identifies "overcredit bias" (false positives on plausible-but-wrong steps) as the dominant PRM failure mode and fixes it with contrastive ranking plus hard negatives. |
| PROGRS: LLM Reasoning with Process Rewards for Outcome-Guided Steps | 2604.02341 [P] | 2026 | Center PRM scores within outcome groups so process signals rank trajectories without overpowering outcome correctness. |
| Probing Entropy Traces for LLM Monitoring | 2601.09001 [P] | 2026 | Uses entropy traces as a monitoring signal across generation; adjacent to step-level probes but token-entropy-based. |

**Takeaway:** the field has already shown that *hidden-state* probes beat scalar PRM
confidence for step correctness (ReProbe, 2504.05419) and that naive PRM scores are
overconfident (2506.09338). The open question is not "can hidden states predict step
correctness" but *which internal object does it best* — exactly the pilot's question,
now transportable to trajectories.

---

## 3. Uncertainty propagation across multi-step reasoning

| Paper | arXiv / venue | Year | One-sentence finding |
|---|---|---|---|
| UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step Agentic Decision-Making | 2506.17419 [V] | 2025 | Decomposes sequential decision uncertainty into intrinsic and extrinsic (mutual-information) parts; extrinsic uncertainty dominates by step 3–4 and UProp improves AUROC up to 9.2% on AgentBench-OS. |
| SAUP: Situation Awareness Uncertainty Propagation on LLM Agent | 2412.01033 [V] (ACL 2025) | 2024 | Propagates per-step uncertainty with situation-aware weights, improving AUROC up to 20% over final-step estimators. |
| Recurrent Confidence Chain: Temporal-Aware Uncertainty Quantification in LLMs | 2601.13368 [V] | 2026 | Inter-step attention plus a hidden confidence state prevents early low-confidence steps from being masked by inflated final confidence. |
| From Sequence to Structure: Relational Uncertainty Propagation for LLM Agents (RUPA) | 2608.16002 [V] | 2026 | Represents a trajectory as a directed graph of reasoning/tool/environment nodes and propagates uncertainty over dependency edges, beating local-signal UQ on τ², Terminal-Bench-2, and GAIA. |
| Every Response Counts: Quantifying Uncertainty of LLM-based Multi-Agent Systems through Tensor Decomposition (MATU) | 2604.08708 [P] | 2026 | Tensor-decomposes multi-agent response structure to capture tool use, multi-step reasoning, and inter-agent communication uncertainty jointly. |
| CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought | 2502.17214 [P] | 2025 | Conditions response-level UQ on the model's own CoT to improve calibration. |
| SelaUr: Self-Evolving LLM Agent via Uncertainty-Aware Rewards | 2602.21158 [U] | 2026 | Uses uncertainty-aware rewards to evolve agent behavior (seen only in citation lists). |
| Thesis Proposal: When Does an Agent Know It Is Lost? Confidence Trajectory Analysis for Tool-Using LLMs | ACL SRW 2026 ([ACL](https://aclanthology.org/2026.acl-srw.78/)) [V] | 2026 | Proposes typed (planning / tool selection / argument / observation) confidence trajectories and argues flat per-step UQ misses action-type-specific failure modes; **no arXiv ID located**. |

**Takeaway:** propagation is well covered at the *aggregation* level (PMI, weights,
graphs). Almost all of it consumes externally observable per-step scores; feeding
*internal-state* per-step probes into these aggregators is an explicit open combination
(2602.05073 §"modeling uncertainty dynamics"; 2609.07395 separately calls out the
step→trajectory gap).

---

## 4. Action-level confidence and tool-use error detection

| Paper | arXiv / venue | Year | One-sentence finding |
|---|---|---|---|
| LLM Agents Already Know When to Call Tools — Even Without Reasoning | 2605.09252 [V] | 2026 | Tool necessity is linearly decodable from the pre-generation hidden state (AUROC 0.89–0.96 across six models), far exceeding the model's verbalized reasoning; a probe-steered prefill cuts tool calls 48% at 1.7% accuracy loss. |
| Critic Experience Bank: Self-Evolving Step-Level Confidence Estimation for LLM Agents | 2607.12397 [P] | 2026 | Frames agent confidence as *pre-execution action productivity*; a training-free critic retrieving past state–action outcomes beats token-uncertainty baselines on web/mobile/shell tasks. |
| Internal Representations as Indicators of Hallucinations in Agent Tool Selection | 2601.05214 [V] | 2026 | Detects tool-calling hallucinations (wrong tool, malformed args, tool bypass) from the last-layer representation of the same forward pass, up to 86.4% accuracy with negligible overhead. |
| Exploring Agentic Tool-Calling Decisions via Uncertainty-Aligned Reinforcement Learning (TRUST) | 2606.06976 [P] | 2026 | Decision-oriented RL destroys the uncertainty separation between correct and incorrect tool calls (wrong-but-low-uncertainty rises 34.5%→70.2%); a UQ-aligned reward restores it while improving decisions. |
| When2Call: When (not) to Call Tools | 2504.18851 [P] | 2025 | Benchmark that makes tool-call necessity a four-way decision (call / ask / abstain / answer directly); SOTA models leave substantial headroom. |
| ParaRecover: A Process-Level Benchmark for Error Localization and Recovery in Parallel Tool-Use Agents | 2609.12345 [P] | 2026 | 10,626 instances over 14 error types test whether agents can localize and recover from intermediate tool errors, where cascade failures are common. |
| Conformal Constrained Policy Optimization for Cost-Effective LLM Agents | 2511.11828 [P] | 2025 | Uses online conformal prediction to choose actions with a reliability guarantee, cutting multi-hop QA cost up to 30%. |
| ToolChain-CRC: Conformal Risk Control for Agentic AI Under Retrieval and Tool-Use Drift | 2606.18467 [P] | 2026 | Calibrates accept/intervene rules at the trajectory level (step risk → trajectory risk → anytime alarm); final-answer-only calibration misses retrieval and tool failures. |

**Takeaway:** the action-level question has split into two camps — post-hoc critics/UQ
(CEB, TRUST, ToolChain-CRC) and internal-state probes (2605.09252, 2601.05214). The
probes report large gains and no training, which is the strongest argument for extending
the team's probe work from answers to actions.

---

## 5. Internal-state probing for agent trajectories

| Paper | arXiv / venue | Year | One-sentence finding |
|---|---|---|---|
| Do Agents Know When They Succeed? Calibrating Agent Confidence from Internal Representations | 2609.09448 [V] | 2026 | Latent Trajectory Dynamics + Action Representation Probe on Qwen2.5-Coder-7B/14B give a zero-overhead reliability monitor that beats surface and sequence baselines. |
| Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade | 2607.06503 [V] | 2026 | Linear probes on internal activations on Qwen2.5-7B (TextCraft/WebShop) predict eventual failure from round 1 and save 1.5–8.8× more compute than single-gate baselines at 90% recall. |
| Detecting Harmful Trajectories from LLM Internal States | 2609.33039 [P] | 2026 | Internal representations support detecting harm in agent trajectories, including tool-calling trajectories. |
| Real-Time Detection and Repair of LLM Agent Failures | 2608.02464 [P] | 2026 | Reports first-round internal probes predicting eventual failure and real-time repair in dialogs. |
| AgentLens: Interpretable Safety Steering via Mechanistic Interpretability | 2606.22673 [P] | 2026 | Uses mechanistic internal signals to detect and mitigate unsafe agent behavior (~96.8% accuracy claimed). |
| UQ taxonomy / TC-ECE evaluation protocol | 2609.07395 [V] | 2026 | Provides the reporting protocol (position-stratified, checkpointed calibration) that any internal-probe agent experiment should adopt. |

**Takeaway:** this is the newest and thinnest thread, and it is dominated by
collaborative evidence for one claim: *agent failure is linearly decodable from hidden
states long before it is visible in behavior* (2607.06503, 2609.09448, 2605.09252,
2601.05214). No paper in this scan probes MLP gate geometry, interaction features, or
answer-direction alignment for agent actions (negative search result; not an exhaustive
novelty audit).

---

## 6. Concrete experiment ideas for `probe_conf`

Ordered by cost; (1)–(3) reuse the existing Qwen2.5 stack and labels.

1. **Replicate ARP/LTD on the project's Qwen2.5 and domains.**
   Run Qwen2.5-7B (and 0.5B/14B if available) in a small tool loop (Bash/SQL/Python as
   in 2609.09448; TextCraft/WebShop as in 2607.06503). Cache hidden states at
   (a) the pre-action decision token and (b) the post-observation token. Train linear
   probes for next-action validity and eventual success. Baselines: token log-probs,
   verbalized confidence, last-step-only probe. This is the minimal bridge from the
   factual pilot to agents and a direct comparison against two Qwen2.5 papers.

2. **Difficulty-controlled agent confidence (the pilot's bins, ported).**
   Define surface difficulty for a trajectory (step index, number of tools, retrieval hit
   rate, argument length). Within difficulty bins, compare probe AUROC against a
   difficulty-only predictor, and correlate probe score with the residual of observed
   success minus the surface prediction. This mirrors the existing PopQA bins/residual
   logic and tests whether internal agent signals survive the difficulty confound that
   verbalized confidence fails. No paper found doing this for agent trajectories
   [negative search result].

3. **MLP interaction features at action boundaries (the project's niche).**
   At the pre-action token, extract gate pre-activations `W1 x`, gates `silu(W1 x)`,
   gated values `silu(W1 x)*(W3 x)`, their entropy/sparsity/top-k mass, and alignment of
   the gated write with candidate action-token unembeddings. Compare against the
   hidden-state probe from (1). Nearest published neighbors are residual-stream ARP
   (2609.09448) and last-layer classifiers (2601.05214); gate-geometry features for
   agent actions appear unclaimed.

4. **Propagation ablation: independent steps vs. UProp/RCC/RUPA aggregation.**
   Take per-step probe probabilities, calibrate them (temperature/isotonic), then compare
   aggregators: last step, mean, UProp PMI (2506.17419), RCC recurrence (2601.13368),
   RUPA graph (2608.16002). Metrics: trajectory AUROC and TC-ECE (2609.07395) with
   position stratification. This tests the step→trajectory gap directly in one model
   family.

5. **Tool-type dichotomy test (evidence vs. verification).**
   Split steps into retrieval/evidence tools and deterministic verification tools
   (code/calculator). Measure probe ECE and verbalized ECE per type. Prediction from
   2601.07264: verbalized confidence is severely overconfident on evidence tools and
   better on verification tools; check whether an internal probe is uniform across types
   or inherits the same asymmetry. This connects the factual-recall probe to tool noise.

6. **Probe-gated tool necessity / abstention.**
   On When2Tool-style tasks (2605.09252) and When2Call (2504.18851), probe the
   pre-generation state for "tool needed" vs. "answer directly." Use the score as a gate
   and measure the accuracy cost at 25–50% tool-call reduction, versus prompt-only and
   log-prob gates. Tests whether the pilot's confidence signal can substitute for a
   steering sentence in Qwen2.5.

7. **Early-warning confidence trajectories with typed actions.**
   Build typed per-step confidence sequences (retrieval / planning / argument / synthesis
   as in the ACL SRW 2026 proposal), then test whether probe-based trajectories flag
   failure earlier than logit-based ones (recall-controlled abort cascade per 2607.06503).
   A useful by-product: compute saved at 90% recall, comparable to the published Qwen2.5
   numbers.

8. **Optional: calibration-aware training comparison.**
   If probes prove useful, compare the plug-in probe against process-margin RL (2604.23333)
   for trajectory calibration. Higher cost; only worth it after (1)–(4) establish signal.

---

## 7. Caveats

- "No paper found" statements are negative search results from this scan, not novelty
  guarantees; 2026 venues are still filling in.
- [P]/[U] rows were not verified on their abs pages; confirm IDs before citing.
  2602.21158, 2604.05952, and 2601.09001 are the least verified.
- ProbeCal (EMNLP 2024) and the ICML 2025 position paper do not currently resolve to
  arXiv IDs; cite the ACL/PMLR pages.
- Several 2025–2026 papers report large gains from internal probes, but sample sizes and
  benchmark diversity vary; ReProbe and 2607.06503 are the most rigorous on methodology
  and calibration, and both include Qwen.
