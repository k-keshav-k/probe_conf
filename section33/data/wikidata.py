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
            payload = response.json()
            break
        except requests.RequestException:
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


def fetch_cohort(
    out_path: Path,
    relations: tuple[str, ...] | None = None,
    per_relation: int = 3000,
    page_size: int = 2000,
    endpoint: str = WIKIDATA_ENDPOINT,
) -> Path:
    """Fetch the requested relations in pages and write a CSV cohort table."""

    relations = relations or tuple(RELATION_QIDS)
    frames = []
    for relation in relations:
        chunks = []
        collected = 0
        offset = 0
        while collected < per_relation:
            page = fetch_relation(relation, limit=page_size, offset=offset, endpoint=endpoint)
            if page.empty:
                break
            chunks.append(page)
            collected += len(page)
            if len(page) < page_size:
                break
            offset += page_size
        frames.append(pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame())
    combined = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    return out_path


def attach_popularity(cohort: pd.DataFrame, pageview_csv: Path, subject_column: str = "subject_id") -> pd.DataFrame:
    """Join a frozen pageview table (subject_id, popularity)."""

    views = pd.read_csv(pageview_csv)
    return cohort.merge(views, on=subject_column, how="left")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch a fresh Wikidata cohort for section33")
    parser.add_argument("--out", type=Path, required=True, help="output CSV path")
    parser.add_argument("--relations", nargs="*", default=None)
    parser.add_argument(
        "--per-relation", type=int, default=3000,
        help="target rows per relation (paged, with retries)",
    )
    parser.add_argument(
        "--endpoint", default=WIKIDATA_ENDPOINT,
        help="SPARQL endpoint (defaults to $WIKIDATA_ENDPOINT or the Wikidata service)",
    )
    args = parser.parse_args(argv)
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
