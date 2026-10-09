"""Wikidata cohort construction helpers (plan section 4.2).

The cluster job fetches a fresh person-question cohort.  This module provides
the documented statement sources, relations, and a thin helper around the
Wikidata Query Service.  It requires the optional ``requests`` dependency and
network access; smoke tests never call it.

Query design:

- one subject per row, four relations: P22 (father), P25 (mother), P19 (place
  of birth), P106 (occupation);
- subjects are humans (P31 = Q5) with an English Wikipedia sitelink;
- page views are joined from a frozen pageview dump outside this file, because
  the Query Service does not serve pageview counts;
- source revisions come from the statement IDs in the response.

The statement pattern is ``?subject p:{property} ?statement .
?statement ps:{property} ?value``.  The earlier ``wdt:`` first hop bound the
*value*, not the statement node, so it returned zero rows while forcing an
expensive join.  Results are fetched in pages with retries for the Query
Service's transient 429/5xx responses.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd

from ..config import RELATION_QIDS, RELATION_QUESTIONS

WIKIDATA_ENDPOINT = os.environ.get(
    "WIKIDATA_ENDPOINT", "https://query.wikidata.org/sparql"
)
USER_AGENT = "section33-cohort/0.1 (research; contact: local user)"

QUERY_TEMPLATE = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT ?subject ?subjectLabel ?statement ?value ?valueLabel ?article ?sitelinks WHERE {{
  ?subject wdt:P31 wd:Q5 ;
           p:{property} ?statement .
  ?statement ps:{property} ?value .
  ?article schema:about ?subject ;
           schema:isPartOf <https://en.wikipedia.org/> .
  ?subject rdfs:label ?subjectLabel . FILTER(LANG(?subjectLabel) = "en")
  ?value rdfs:label ?valueLabel . FILTER(LANG(?valueLabel) = "en")
  OPTIONAL {{ ?subject wikibase:sitelinks ?sitelinks }}
}}
LIMIT {limit}
OFFSET {offset}
"""


def fetch_relation(
    relation: str,
    limit: int = 2000,
    offset: int = 0,
    endpoint: str = WIKIDATA_ENDPOINT,
    retries: int = 4,
    backoff: float = 5.0,
) -> pd.DataFrame:
    """Fetch one page of statement rows, retrying transient Query Service errors."""

    import requests

    prop = RELATION_QIDS[relation]
    query = QUERY_TEMPLATE.format(property=prop, limit=limit, offset=offset)
    for attempt in range(retries + 1):
        try:
            response = requests.get(
                endpoint,
                params={"query": query, "format": "json"},
                headers={"User-Agent": USER_AGENT},
                timeout=120,
            )
            response.raise_for_status()
            payload = json.loads(response.text, strict=False)
            break
        except (requests.RequestException, ValueError):
            if attempt == retries:
                raise
            time.sleep(backoff * (2**attempt))
    rows = []
    template = RELATION_QUESTIONS[relation]
    for binding in payload["results"]["bindings"]:
        statement = binding.get("statement", {}).get("value", "")
        subject_id = binding["subject"]["value"].rsplit("/", 1)[-1]
        subject_name = binding.get("subjectLabel", {}).get("value", "") or subject_id
        rows.append(
            {
                "question_id": statement.rsplit("/", 1)[-1] if statement else f"{subject_id}-{prop}",
                "subject_id": subject_id,
                "subject_name": subject_name,
                "relation": relation,
                "relation_qid": prop,
                "question": template.format(subject=subject_name),
                "statement": statement,
                "article_url": binding.get("article", {}).get("value", ""),
                "sitelinks": int(binding["sitelinks"]["value"])
                if "sitelinks" in binding
                else None,
                "reference_answers": binding.get("valueLabel", {}).get("value", ""),
                "source_revision": statement,
            }
        )
    return pd.DataFrame(rows)


def _fetch_page(relation: str, offset: int, limit: int, endpoint: str):
    """Fetch one page, halving the page size when a response is unusable.

    The Query Service occasionally returns truncated JSON for large pages at
    deep offsets; smaller pages at the same offset are reliable.
    """

    while True:
        try:
            page = fetch_relation(relation, limit=limit, offset=offset, endpoint=endpoint)
            return page, limit
        except (OSError, ValueError):
            if limit <= 200:
                raise
            limit //= 2


def fetch_cohort(
    out_path: Path,
    relations: tuple[str, ...] | None = None,
    per_relation: int = 10_000,
    page_size: int = 1000,
    endpoint: str = WIKIDATA_ENDPOINT,
) -> Path:
    """Fetch the requested relations in pages and write a CSV cohort table.

    Small pages avoid the Query Service truncating large deep-offset responses;
    the row target is generous because several statements can share one subject
    (occupation especially), so unique-subject counts are lower than row counts.
    """

    relations = relations or tuple(RELATION_QIDS)
    frames = []
    for relation in relations:
        chunks = []
        collected = 0
        offset = 0
        while collected < per_relation:
            page, step = _fetch_page(relation, offset, page_size, endpoint)
            if page.empty:
                break
            chunks.append(page)
            collected += len(page)
            print(f"{relation}: {collected} rows fetched", flush=True)
            if len(page) < step:
                break
            offset += step
        frames.append(pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame())
    combined = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    return out_path


def attach_popularity(cohort: pd.DataFrame, pageview_csv: Path, subject_column: str = "subject_id") -> pd.DataFrame:
    """Join a frozen pageview table (subject_id, popularity)."""

    views = pd.read_csv(pageview_csv)
    return cohort.merge(views, on=subject_column, how="left")


SITELINK_QUERY = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT ?subject ?sitelinks WHERE {{
  VALUES ?subject {{ {values} }}
  ?subject wikibase:sitelinks ?sitelinks .
}}
"""


def fetch_sitelinks(
    subject_ids,
    endpoint: str = WIKIDATA_ENDPOINT,
    chunk_size: int = 400,
    retries: int = 3,
    backoff: float = 2.0,
) -> dict[str, float]:
    """Bulk sitelink counts (number of Wikipedia language editions)."""

    import requests

    result: dict[str, float] = {}
    ids = list(dict.fromkeys(str(subject) for subject in subject_ids))
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        values = " ".join(f"wd:{qid}" for qid in chunk)
        query = SITELINK_QUERY.format(values=values)
        for attempt in range(retries + 1):
            try:
                response = requests.get(
                    endpoint,
                    params={"query": query, "format": "json"},
                    headers={"User-Agent": USER_AGENT},
                    timeout=60,
                )
                response.raise_for_status()
                for binding in json.loads(response.text, strict=False)["results"]["bindings"]:
                    qid = binding["subject"]["value"].rsplit("/", 1)[-1]
                    result[qid] = float(binding["sitelinks"]["value"])
                break
            except (requests.RequestException, ValueError):
                if attempt == retries:
                    raise
                time.sleep(backoff * (2**attempt))
    return result


def fill_popularity(
    cohort_csv: Path,
    pageviews_csv: Path | None = None,
    endpoint: str = WIKIDATA_ENDPOINT,
) -> pd.DataFrame:
    """Fill missing popularity in an existing cohort CSV, in place.

    Reuses whatever is already there: stored pageview values win, the stored
    ``sitelinks`` column fills the rest, and a bulk SPARQL lookup covers any
    remaining subjects.  No full cohort refetch is needed.
    """

    path = Path(cohort_csv)
    frame = pd.read_csv(path)
    if "popularity" not in frame.columns:
        frame["popularity"] = float("nan")
    frame["popularity"] = pd.to_numeric(frame["popularity"], errors="coerce")

    if pageviews_csv is not None and Path(pageviews_csv).exists():
        views = pd.read_csv(pageviews_csv)
        if "popularity" in views.columns and "subject_id" in views.columns:
            views = views[["subject_id", "popularity"]].rename(columns={"popularity": "views"})
            frame = frame.merge(views, on="subject_id", how="left")
            frame["popularity"] = frame["popularity"].fillna(frame["views"])
            frame = frame.drop(columns="views")

    if "sitelinks" in frame.columns:
        frame["popularity"] = frame["popularity"].fillna(
            pd.to_numeric(frame["sitelinks"], errors="coerce")
        )

    missing = frame["popularity"].isna()
    if missing.any():
        lookup = fetch_sitelinks(
            frame.loc[missing, "subject_id"].astype(str).unique(), endpoint=endpoint
        )
        frame.loc[missing, "popularity"] = frame.loc[missing, "subject_id"].astype(str).map(lookup)

    frame.to_csv(path, index=False)
    return frame


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch a fresh Wikidata cohort for section33")
    parser.add_argument("--out", type=Path, required=True, help="output CSV path")
    parser.add_argument("--relations", nargs="*", default=None)
    parser.add_argument(
        "--per-relation", type=int, default=10_000,
        help="target rows per relation (paged; unique subjects are fewer)",
    )
    parser.add_argument(
        "--endpoint", default=WIKIDATA_ENDPOINT,
        help="SPARQL endpoint (defaults to $WIKIDATA_ENDPOINT or the Wikidata service)",
    )
    parser.add_argument(
        "--fill-popularity", action="store_true",
        help="fill missing popularity in --out from pageviews and sitelinks, then exit",
    )
    parser.add_argument(
        "--pageviews", type=Path, default=None,
        help="optional subject_id,popularity CSV used before sitelinks",
    )
    args = parser.parse_args(argv)
    if args.fill_popularity:
        frame = fill_popularity(args.out, pageviews_csv=args.pageviews, endpoint=args.endpoint)
        print(
            f"popularity coverage: {int(frame['popularity'].notna().sum())} of {len(frame)} rows"
        )
        return 0
    path = fetch_cohort(
        args.out,
        tuple(args.relations) if args.relations else None,
        per_relation=args.per_relation,
        endpoint=args.endpoint,
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
