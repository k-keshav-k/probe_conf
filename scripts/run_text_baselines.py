"""Run the input-only text baselines for the retrieval-monitoring plan.

Usage:

    python scripts/run_text_baselines.py --run runs/section33_confirmatory --track B --track A
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from section33.analysis.text_baselines import run_track, save_reports  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/section33_confirmatory"))
    parser.add_argument("--track", action="append", default=None)
    args = parser.parse_args()
    tracks = args.track or ["A", "B"]
    all_reports = []
    all_scores = []
    for track in tracks:
        modes = ("question",) if track == "A" else ("question", "question_candidate")
        reports, scores = run_track(args.run, track, modes)
        all_reports.extend(reports)
        all_scores.append(scores)
        for report in reports:
            within = report.within_question_auroc
            within_text = "n/a" if within is None else f"{within:.3f}"
            print(
                f"track {report.track} mode {report.mode}: "
                f"overall {report.overall_auroc:.3f}, "
                f"within-question {within_text} "
                f"({report.eligible_questions} mixed questions), "
                f"saved H {report.saved['saved_H']:.3f}, "
                f"delta vs H {report.delta_vs_hidden:+.3f} "
                f"[{report.delta_ci[0]:+.3f}, {report.delta_ci[1]:+.3f}]"
            )
    import pandas as pd

    path = save_reports(
        args.run, all_reports, pd.concat(all_scores, ignore_index=True)
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
