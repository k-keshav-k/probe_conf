"""Fetch monthly Wikipedia pageviews for the fresh cohort (plan section 4.2).

The Wikidata query stores each subject's English Wikipedia article URL.  This
module queries the Wikimedia Analytics pageviews API per article and writes a
frozen ``subject_id, popularity`` table that ``attach_popularity`` joins into
the cohort CSV.  The window is the trailing 12 completed months by default and
is recorded in the output for provenance.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd

PAGEVIEW_ENDPOINT = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{article}/monthly/{start}/{end}"
)
USER_AGENT = "section33-cohort/0.1 (research; contact: local user)"


def article_title(url: str) -> str | None:
    """Extract the enwiki article title from a Wikidata sitelink URL."""

    if not url or "/wiki/" not in str(url):
        return None
    return unquote(str(url).split("/wiki/", 1)[1])


def default_window(today: date | None = None) -> tuple[str, str]:
    """Trailing 12 completed months, as ``YYYYMMDD`` bounds for the API."""

    today = today or date.today()
    end_year, end_month = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    start_year, start_month = (
        (end_year - 1, end_month + 1) if end_month < 12 else (end_year - 1, 1)
    )
    return f"{start_year}{start_month:02d}01", f"{end_year}{end_month:02d}01"


def fetch_views(
    article: str,
    start: str,
    end: str,
    retries: int = 3,
    backoff: float = 1.0,
) -> float | None:
    """Total monthly views for one article, or None when there is no data."""

    import requests

    url = PAGEVIEW_ENDPOINT.format(article=quote(article, safe=""), start=start, end=end)
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            items = response.json().get("items", [])
            if not items:
                return None
            return float(sum(item.get("views", 0) for item in items))
        except requests.RequestException:
            if attempt == retries:
                return None
            time.sleep(backoff * (2**attempt))
    return None


def fetch_cohort_pageviews(
    cohort_csv: Path,
    out_csv: Path,
    start: str,
    end: str,
    workers: int = 8,
    retries: int = 3,
) -> pd.DataFrame:
    """Fetch pageviews for every distinct subject and write the join table."""

    cohort = pd.read_csv(cohort_csv)
    subjects = cohort.drop_duplicates("subject_id")
    if "article_url" in subjects.columns:
        titles = subjects.set_index("subject_id")["article_url"].map(article_title)
    else:
        titles = subjects.set_index("subject_id")["subject_name"].map(
            lambda name: str(name).replace(" ", "_")
        )
    titles = titles.dropna()

    results: dict[str, float | None] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_views, title, start, end, retries): subject_id
            for subject_id, title in titles.items()
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    frame = pd.DataFrame(
        {
            "subject_id": list(results),
            "popularity": [results[subject_id] for subject_id in results],
            "pageview_start": start,
            "pageview_end": end,
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    return frame


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch monthly pageviews for a cohort")
    parser.add_argument("--cohort", type=Path, required=True, help="cohort CSV with article_url")
    parser.add_argument("--out", type=Path, required=True, help="output subject_id,popularity CSV")
    parser.add_argument("--start", default=None, help="YYYYMMDD (default: trailing 12 months)")
    parser.add_argument("--end", default=None, help="YYYYMMDD (default: last completed month)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)
    start, end = (args.start, args.end) if args.start and args.end else default_window()
    frame = fetch_cohort_pageviews(
        args.cohort, args.out, start, end, workers=args.workers, retries=args.retries
    )
    print(
        f"wrote {args.out}: {int(frame['popularity'].notna().sum())} of {len(frame)} subjects "
        f"with views ({start}..{end})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
