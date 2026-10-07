"""Plot only the evidence for the MLP-confidence claim in meet.md section 3.3.

Read saved scores and audited paired intervals; do not fit new probes.
Run: .venv/bin/python -B scripts/make_summary_claim_figures.py
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
OUT = ROOT / 'report/figures'
H, M, C = 'H_ans_l18', 'Msum_last_l2', 'H18+Msum_all'
COLORS = {'confidence': '#888888', H: '#3e70a1', M: '#cc783c', C: '#328663'}
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.dpi': 190, 'font.family': 'DejaVu Sans'})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    main_scores = pd.read_csv(RES / 'pilot7b_frac_extra.csv').set_index('feature_set')
    greedy = pd.read_csv(RES / 'pilot7b_greedy_extra.csv').set_index('feature_set')
    audit = json.loads((RES / 'audit_checks.json').read_text())
    keys = ['confidence', M, H, C]
    labels = ['Stated confidence', 'MLP summaries only', 'Hidden state only', 'Hidden state + MLP summaries']
    fig, ax = plt.subplots(figsize=(9.8, 4.1), layout='constrained')
    for i, (key, label) in enumerate(zip(keys, labels)):
        row = main_scores.loc[key]
        ax.errorbar(row.auroc_oof, i, xerr=[[row.auroc_oof-row.ci_lo], [row.ci_hi-row.auroc_oof]],
                    fmt='o', markersize=8, capsize=5, color=COLORS[key], linewidth=2)
        ax.text(.978, i, f'{row.auroc_oof:.3f}', va='center', ha='right', fontweight='bold')
    ax.set_yticks(range(4), labels)
    ax.invert_yaxis()
    ax.set_xlim(.46, .99)
    ax.axvline(.5, color='#888888', linestyle='--', linewidth=1)
    ax.set_xlabel('AUROC for success on at least 15 of 30 answers')
    ax.set_title('Adding MLP summaries gives the highest score', pad=15)
    ax.grid(axis='x', alpha=.2)
    save(fig, 'summary_mlp_scores.png')

    # Intervals come from the existing audit, which replays the saved paired bootstrap.
    specs = [('frac', C, 'Add MLP summaries\n(main target)', main_scores),
             ('greedy', C, 'Add MLP summaries\n(greedy-answer target)', greedy),
             ('frac', M, 'Replace hidden state with MLP\n(main target)', main_scores)]
    fig, ax = plt.subplots(figsize=(9.8, 3.8), layout='constrained')
    for i, (target, key, label, scores) in enumerate(specs):
        pair = audit['paired'][target][f'{key} minus {H}']
        lo, hi = pair['ci']
        delta = scores.loc[key, 'auroc_oof'] - scores.loc[H, 'auroc_oof']
        ax.errorbar(delta, i, xerr=[[delta-lo], [hi-delta]], fmt='o', capsize=5,
                    markersize=8, color=COLORS[key], linewidth=2)
        ax.text(.079, i, f'{delta:+.3f}', va='center', ha='right', fontweight='bold')
    ax.set_yticks(range(3), [s[2] for s in specs])
    ax.set_ylim(2.5, -.5)
    ax.set_xlim(-.05, .085)
    ax.axvline(0, color='#444444', linestyle='--', linewidth=1.2)
    ax.set_xlabel('AUROC difference from the hidden-state probe')
    ax.set_title('Adding MLP features helps; replacing hidden states is inconclusive', pad=15)
    ax.grid(axis='x', alpha=.2)
    save(fig, 'summary_mlp_differences.png')

    bins = pd.read_csv(RES / 'pilot7b_frac_extra_withinbin.csv').set_index('score')
    fig, ax = plt.subplots(figsize=(9.8, 4.3), layout='constrained')
    for key, label, marker in [(M, 'MLP summaries only', 's'), (H, 'Hidden state only', 'o'),
                               (C, 'Hidden state + MLP summaries', 'D')]:
        ax.plot(range(5), bins.loc[key, [f'bin{i}' for i in range(5)]].astype(float),
                marker=marker, linewidth=2, markersize=7, label=label, color=COLORS[key])
    ax.set_xticks(range(5), ['Q1\nLeast popular', 'Q2', 'Q3', 'Q4', 'Q5\nMost popular'])
    ax.set_ylim(.75, .96)
    ax.set_ylabel('AUROC within each popularity group')
    ax.set_title('The combined probe scores higher in all five groups', pad=15)
    ax.grid(alpha=.2)
    ax.legend(loc='lower center', bbox_to_anchor=(.5, -.37), ncol=2, frameon=False, fontsize=10)
    save(fig, 'summary_mlp_popularity.png')
    print('Wrote three MLP-claim plots from saved evidence.')


if __name__ == '__main__':
    main()
