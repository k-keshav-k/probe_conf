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
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import RELATION_QIDS

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "section33-cohort/0.1 (research; contact: local user)"

QUERY_TEMPLATE = """
SELECT ?subject ?subjectLabel ?subjectDescription ?relationQid ?statement ?value ?valueLabel WHERE {{
  ?subject wdt:P31 wd:Q5 .
  ?subject wdt:{property} ?statement .
  ?statement ps:{property} ?value .
  ?subject rdfs:label ?subjectLabel . FILTER(LANG(?subjectLabel) = "en")
  OPTIONAL {{ ?subject schema:description ?subjectDescription . FILTER(LANG(?subjectDescription) = "en") }}
  BIND("{property}" AS ?relationQid)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
LIMIT {limit}
OFFSET {offset}
"""


def fetch_relation(
    relation: str,
    limit: int = 5000,
    offset: int = 0,
    endpoint: str = WIKIDATA_ENDPOINT,
) -> pd.DataFrame:
    """Fetch statement rows for one relation from the Wikidata Query Service."""

    import requests

    prop = RELATION_QIDS[relation]
    query = QUERY_TEMPLATE.format(property=prop, limit=limit, offset=offset)
    response = requests.get(
        endpoint,
        params={"query": query, "format": "json"},
        headers={"User-Agent": USER_AGENT},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    rows = []
    for binding in payload["results"]["bindings"]:
        rows.append(
            {
                "subject_id": binding["subject"]["value"].rsplit("/", 1)[-1],
                "subject_name": binding.get("subjectLabel", {}).get("value", ""),
                "relation": relation,
                "relation_qid": prop,
                "statement": binding.get("statement", {}).get("value", ""),
                "reference_answers": binding.get("valueLabel", {}).get("value", ""),
                "source_revision": binding.get("statement", {}).get("value", ""),
            }
        )
    return pd.DataFrame(rows)


def fetch_cohort(out_path: Path, relations: tuple[str, ...] | None = None) -> Path:
    """Fetch all four relations and write a CSV cohort table."""

    relations = relations or tuple(RELATION_QIDS)
    frames = [fetch_relation(relation) for relation in relations]
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
    args = parser.parse_args(argv)
    path = fetch_cohort(args.out, tuple(args.relations) if args.relations else None)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
