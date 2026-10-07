# Pilot probe report (`fs15_greedy`)

Historical small-model pipeline check. These numbers do not establish the main
research claims. Gold dependencies and evaluation limits are described in
`../report/AUDIT.md`.

- questions: 60
- positive rate (label): 0.117
- greedy accuracy: 0.117
- mean verbalized confidence: 92.1

## Headline AUROCs (pooled out-of-fold)

| feature family | best AUROC | 95% CI | best variant |
| --- | --- | --- | --- |
| surface (difficulty) | 0.790 | [0.656, 0.914] | `surface` |
| model probs | 0.407 | [0.222, 0.590] | `prob` |
| verbalized confidence | 0.423 | [0.175, 0.657] | `confidence` |
| hidden states | 0.668 | [0.433, 0.887] | `H_ans_l15` |
| MLP summaries | 0.811 | [0.642, 0.960] | `Msum_ans_l24` |

## Verbalized confidence vs difficulty

- Spearman(confidence, log10 s_pop) = -0.072
- Spearman(confidence, frac_correct) = 0.152
- Spearman(confidence, p_gold) = 0.136
- AUROC(confidence -> correct) = 0.423 (compared with probes above)

## Layer curves

![layer curves](figures/fs15_greedy_layer_curves.png)
![best families](figures/fs15_greedy_best_families.png)
![diagnostics](figures/fs15_greedy_diagnostics.png)

## Verdict against hypotheses (auto-computed, to be interpreted)

- H1 (MLP signal > surface): MLP 0.811 vs surface 0.790 -> supported
- H2 (comparable to hidden): delta = +0.143 -> not supported


## Stage 2: difficulty control (s_pop quintiles)

| score                 | feature_set   |   overall_auroc |   bin0_auroc |   bin1_auroc |   bin2_auroc |   bin3_auroc |   bin4_auroc |
|:----------------------|:--------------|----------------:|-------------:|-------------:|-------------:|-------------:|-------------:|
| surface (difficulty)  | surface       |           0.790 |          nan |        1.000 |        0.844 |        0.818 |        0.545 |
| verbalized confidence | confidence    |           0.423 |          nan |        0.227 |        0.516 |        0.136 |        0.409 |
| model probs           | prob          |           0.407 |          nan |        0.364 |        0.344 |        0.182 |        0.545 |
| hidden states         | H_ans_l15     |           0.668 |          nan |        1.000 |        0.906 |        0.364 |        0.364 |
| MLP summaries         | Msum_ans_l24  |           0.811 |          nan |        0.909 |        0.875 |        1.000 |        0.818 |

### Per-relation AUROC

```
                score  father  mother  occupation  place of birth
 surface (difficulty)     NaN   0.786       0.833           0.708
verbalized confidence     NaN   0.429       0.299           0.583
          model probs     NaN   0.571       0.333           0.667
        hidden states     NaN   0.357       0.736           0.625
        MLP summaries     NaN   0.857       0.681           0.792
```

Residual diagnostic (Spearman with `frac_correct - surface_OOF`):

| score                 |   spearman_with_residual |
|:----------------------|-------------------------:|
| surface (difficulty)  |                   -0.586 |
| verbalized confidence |                    0.006 |
| model probs           |                    0.035 |
| hidden states         |                   -0.177 |
| MLP summaries         |                   -0.064 |

- Paired bootstrap delta (MLP summaries - hidden states): +0.144 [-0.104, +0.390]
- Permutation check (`H_ans_l14`, shuffled labels): AUROC 0.485 (expected ~0.5)

*Caveats: best layer/variant selected on pooled OOF scores (optimistic bias); single model, single dataset; K-sample labels are noisy; answer-alignment features use the gold token (oracle) and are not deployable as-is.*
