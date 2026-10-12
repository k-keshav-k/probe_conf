"""Fetch pilot subjects that have two relations at once.

The raw cohort source fetches each relation independently, so very few
subjects appear in two relations.  The S3 pilot needs subjects with two facts
(for example father and occupation), so this module queries Wikidata for
subjects that have both and writes a pairs CSV.

Run on the login node (compute nodes have no internet):

    python -m retrieval_monitoring.cli fetch-pairs \
        --out "$RUN_DIR/retrieval_monitoring/pilot_pairs.csv" --limit 200
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from section33.config import RELATION_QIDS, RELATION_QUESTIONS

ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "section33-cohort/0.1 (research; contact: local user)"

PAIR_QUERY = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
SELECT ?subject ?subjectLabel ?firstLabel ?secondLabel ?sitelinks WHERE {{
  ?subject wdt:P31 wd:Q5 ;
           wdt:{first} ?first ;
           wdt:{second} ?second ;
           wikibase:sitelinks ?sitelinks .
  FILTER(?sitelinks >= {min_sitelinks})
  ?article schema:about ?subject ;
           schema:isPartOf <https://en.wikipedia.org/> .
  ?subject rdfs:label ?subjectLabel . FILTER(LANG(?subjectLabel) = "en")
  ?first rdfs:label ?firstLabel . FILTER(LANG(?firstLabel) = "en")
  ?second rdfs:label ?secondLabel . FILTER(LANG(?secondLabel) = "en")
}}
LIMIT {limit}
"""


def fetch_pairs(
    relations: tuple[str, str] = ("father", "occupation"),
    limit: int = 200,
    endpoint: str = ENDPOINT,
    retries: int = 4,
    backoff: float = 3.0,
    min_sitelinks: int = 40,
) -> pd.DataFrame:
    """Fetch subjects with both relations, most-linked (famous) first.

    The pilot needs headroom: obscure subjects sit at the success floor even
    clean, so damage cannot be measured.  A sitelinks filter keeps widely known
    subjects; the endpoint sorts them locally because ``ORDER BY`` over the full
    match set times out.
    """

    import requests

    query = PAIR_QUERY.format(
        first=RELATION_QIDS[relations[0]],
        second=RELATION_QIDS[relations[1]],
        limit=max(limit * 4, 400),
        min_sitelinks=min_sitelinks,
    )
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
        except (OSError, ValueError):
            if attempt == retries:
                raise
            time.sleep(backoff * (2**attempt))
    rows = []
    for binding in payload["results"]["bindings"]:
        rows.append(
            {
                "subject_id": binding["subject"]["value"].rsplit("/", 1)[-1],
                "subject_name": binding["subjectLabel"]["value"],
                "sitelinks": int(binding["sitelinks"]["value"]),
                relations[0]: binding["firstLabel"]["value"],
                relations[1]: binding["secondLabel"]["value"],
            }
        )
    frame = (
        pd.DataFrame(rows)
        .drop_duplicates("subject_id")
        .sort_values("sitelinks", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
    return frame


def pairs_to_questions(
    frame: pd.DataFrame, relations: tuple[str, str] = ("father", "occupation")
) -> pd.DataFrame:
    """Long table: one question per subject and relation."""

    rows = []
    for row in frame.itertuples(index=False):
        subject_id = str(getattr(row, "subject_id"))
        name = str(getattr(row, "subject_name"))
        for relation in relations:
            answer = getattr(row, relation)
            rows.append(
                {
                    "question_id": f"pilot-{subject_id}-{relation}",
                    "subject_id": subject_id,
                    "subject_name": name,
                    "relation": relation,
                    "question": RELATION_QUESTIONS[relation].format(subject=name),
                    "reference_answers": str(answer),
                }
            )
    return pd.DataFrame(rows)


def write_pairs_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs_to_questions(frame).to_csv(path, index=False)
    return path


def load_pairs_csv(path: Path, relations: tuple[str, ...]) -> list[dict]:
    """Rebuild the pilot pair list from a pairs CSV."""

    frame = pd.read_csv(path)
    frame["subject_id"] = frame["subject_id"].astype(str)
    pairs = []
    for subject_id, group in frame.groupby("subject_id", sort=False):
        entries = []
        for relation in relations:
            match = group[group["relation"] == relation]
            if match.empty:
                continue
            row = match.iloc[0]
            reference = row["reference_answers"]
            if isinstance(reference, str):
                reference = [item.strip() for item in reference.split("|") if item.strip()]
            else:
                reference = [str(reference)]
            entries.append(
                {
                    "relation": relation,
                    "question_id": str(row["question_id"]),
                    "question": str(row["question"]),
                    "reference_answers": reference,
                }
            )
        if len(entries) == len(relations):
            pairs.append(
                {
                    "subject_id": subject_id,
                    "name": str(group.iloc[0]["subject_name"]),
                    "relations": entries,
                }
            )
    return pairs
