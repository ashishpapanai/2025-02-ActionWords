from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from bs4 import BeautifulSoup
from requests import Response, Session
from tqdm import tqdm

SEARCH_ENDPOINT = "https://search.gesis.org/searchengine"
VARIABLE_DETAIL_URL = "https://search.gesis.org/variables/{variable_id}?lang=en"
STUDY_DETAIL_URL = "https://search.gesis.org/research_data/{study_id}?lang=en"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
MAX_REQUEST_SIZE = 500
REQUEST_TIMEOUT = 30


def _collapse_whitespace(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    if isinstance(text, (list, tuple, set)):
        text = " ".join(str(item) for item in text if item is not None)
    if isinstance(text, (int, float)):
        text = str(text)
    collapsed = " ".join(text.split())
    return collapsed or None


def _parse_html_table(table_html: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not table_html:
        return None

    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if not table:
        return None

    # Extract header labels; fall back to positional names if missing.
    header_section = table.find("thead")
    if header_section and header_section.find("tr"):
        header_cells = header_section.find("tr").find_all("th")
    else:
        first_row = table.find("tr")
        header_cells = first_row.find_all(["th", "td"]) if first_row else []

    headers: List[str]
    if header_cells:
        headers = [_normalise_header(cell.get_text(
            " ", strip=True)) for cell in header_cells]
    else:
        headers = []

    body = table.find("tbody") or table
    rows: List[Dict[str, Any]] = []
    for row in body.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        if headers and len(cells) != len(headers):
            # Some tables repeat header rows inside <tbody>.
            if row.find("th"):
                continue
            # Fall back to positional headers for irregular rows.
            local_headers = [f"column_{idx}" for idx in range(len(cells))]
        else:
            local_headers = headers or [
                f"column_{idx}" for idx in range(len(cells))]
        if len(cells) != len(headers):
            # Already handled via local_headers fallback.
            pass
        row_dict = {
            local_headers[idx]: cells[idx].get_text(" ", strip=True)
            for idx in range(len(cells))
        }
        rows.append(row_dict)

    return rows or None

    def _normalise_header(raw: str) -> str:
        tokens = raw.lower()
        if tokens.startswith("value"):
            return "Value"
        if tokens.startswith("label"):
            return "Label"
        if tokens.startswith("missing"):
            return "Missing"
        if tokens.startswith("count"):
            return "Count"
        if tokens.startswith("valid") or "valid percent" in tokens:
            return "Valid Percent"
        if tokens.startswith("percent"):
            return "Percent"
        return raw.strip() or "column"


@dataclass
class VariableRecord:

    variable_name: str
    question_en: Optional[str]
    question_de: Optional[str]
    title_en: Optional[str]
    title_de: Optional[str]
    variable_id: str
    detail_url: str
    study_id: str
    study_title_en: Optional[str]
    study_title_de: Optional[str]
    topics_en: Optional[List[str]]
    topics_de: Optional[List[str]]
    interview_instructions_en: Optional[str]
    interview_instructions_de: Optional[str]
    time_collection_years: Optional[List[str]]
    notes_en: Optional[str]
    notes_de: Optional[str]
    analysis_unit_en: Optional[str]
    analysis_unit_de: Optional[str]
    kind_of_data_en: Optional[str]
    kind_of_data_de: Optional[str]
    code_list: Optional[List[Dict[str, Any]]]
    frequency_table: Optional[List[Dict[str, Any]]]

    @classmethod
    def from_source(cls, source: Dict[str, Any]) -> "VariableRecord":

        def as_list(value: Any) -> Optional[List[str]]:
            if value is None:
                return None
            if isinstance(value, list):
                return [item for item in map(_collapse_whitespace, value) if item]
            return [_collapse_whitespace(str(value))] if str(value).strip() else None

        return cls(
            variable_name=source.get(
                "variable_name") or source["title"].split(" - ", 1)[0],
            question_en=_collapse_whitespace(
                source.get("variable_label_en")
                or _extract_question_from_title(source.get("title_en"))
            ),
            question_de=_collapse_whitespace(
                source.get("variable_label")
                or _extract_question_from_title(source.get("title"))
            ),
            title_en=_collapse_whitespace(source.get("title_en")),
            title_de=_collapse_whitespace(source.get("title")),
            variable_id=source["id"],
            detail_url=VARIABLE_DETAIL_URL.format(variable_id=source["id"]),
            study_id=source.get("study_id"),
            study_title_en=_collapse_whitespace(source.get("study_title_en")),
            study_title_de=_collapse_whitespace(source.get("study_title")),
            topics_en=as_list(source.get("topic_en")),
            topics_de=as_list(source.get("topic")),
            interview_instructions_en=_collapse_whitespace(
                source.get("variable_interview_instructions_en")
            ),
            interview_instructions_de=_collapse_whitespace(
                source.get("variable_interview_instructions")
            ),
            time_collection_years=as_list(source.get("time_collection_years")),
            notes_en=_collapse_whitespace(source.get("notes_en")),
            notes_de=_collapse_whitespace(source.get("notes")),
            analysis_unit_en=_collapse_whitespace(
                source.get("analysis_unit_en")),
            analysis_unit_de=_collapse_whitespace(source.get("analysis_unit")),
            kind_of_data_en=_collapse_whitespace(source.get("kind_data_en")),
            kind_of_data_de=_collapse_whitespace(source.get("kind_data")),
            code_list=_parse_html_table(source.get("variable_code_list")),
            frequency_table=_parse_html_table(
                source.get("codebook_table_html")),
        )


def _extract_question_from_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return None
    if " - " in title:
        return title.split(" - ", 1)[1].strip()
    return title.strip() or None


def _normalise_header(raw: str) -> str:
    tokens = raw.lower()
    if tokens.startswith("value"):
        return "Value"
    if tokens.startswith("label"):
        return "Label"
    if tokens.startswith("missing"):
        return "Missing"
    if tokens.startswith("count"):
        return "Count"
    if tokens.startswith("valid") or "valid percent" in tokens:
        return "Valid Percent"
    if tokens.startswith("percent"):
        return "Percent"
    return raw.strip() or "column"


def _parse_html_table(table_html: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    if not table_html:
        return None

    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table")
    if not table:
        return None

    header_section = table.find("thead")
    if header_section and header_section.find("tr"):
        header_cells = header_section.find("tr").find_all("th")
    else:
        first_row = table.find("tr")
        header_cells = first_row.find_all(["th", "td"]) if first_row else []

    if header_cells:
        headers = [_normalise_header(cell.get_text(
            " ", strip=True)) for cell in header_cells]
    else:
        headers = []

    body = table.find("tbody") or table
    rows: List[Dict[str, Any]] = []
    for row in body.find_all("tr"):
        cells = row.find_all("td")
        if not cells:
            continue
        if headers and len(cells) != len(headers):
            if row.find("th"):
                continue
            local_headers = [f"column_{idx}" for idx in range(len(cells))]
        else:
            local_headers = headers or [
                f"column_{idx}" for idx in range(len(cells))]
        row_dict = {
            local_headers[idx]: cells[idx].get_text(" ", strip=True)
            for idx in range(len(cells))
        }
        rows.append(row_dict)

    return rows or None


def _fetch_batch(
    session: Session,
    query: str,
    batch_size: int,
    offset: int,
) -> Dict[str, Any]:
    params = {
        "q": query,
        "size": batch_size,
        "from": offset,
        "sort": "variable_order:asc",
    }
    response: Response = session.get(
        SEARCH_ENDPOINT,
        params=params,
        headers=DEFAULT_HEADERS,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def fetch_variables_for_study(study_id: str, page_size: int = 200) -> List[Dict[str, Any]]:
    if page_size <= 0 or page_size > MAX_REQUEST_SIZE:
        raise ValueError(f"page_size must be between 1 and {MAX_REQUEST_SIZE}")

    query = f'type:"variables" AND study_id:"{study_id}"'
    session = requests.Session()
    records: List[Dict[str, Any]] = []
    total: Optional[int] = None
    offset = 0

    with tqdm(desc=f"Fetching variables for {study_id}") as progress:
        while True:
            payload = _fetch_batch(session, query, page_size, offset)
            hits = payload.get("hits", {})
            total_value = hits.get("total", {}).get("value", 0)
            if total is None:
                total = total_value
                progress.total = total
                progress.refresh()

            batch = hits.get("hits", [])
            if not batch:
                break

            for item in batch:
                records.append(item.get("_source", {}))
            offset += len(batch)
            progress.update(len(batch))

            if len(records) >= total_value:
                break

    return records


def build_payload(
    study_id: str,
    raw_variables: Iterable[Dict[str, Any]],
    include_raw: bool = False,


) -> Dict[str, Any]:
    structured_variables: Dict[str, Dict[str, Any]] = {}
    study_title_en: Optional[str] = None
    study_title_de: Optional[str] = None

    for source in raw_variables:
        record = VariableRecord.from_source(source)
        if not study_title_en:
            study_title_en = record.study_title_en
        if not study_title_de:
            study_title_de = record.study_title_de

        record_dict = asdict(record)
        structured_variables[record.variable_name] = record_dict

        if include_raw:
            structured_variables[record.variable_name]["_raw"] = source

    return {
        "study_id": study_id,
        "study_title_en": study_title_en,
        "study_title_de": study_title_de,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "dataset_url": STUDY_DETAIL_URL.format(study_id=study_id),
        "variable_count": len(structured_variables),
        "variables": structured_variables,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GESIS variable questions to JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("study_id", help="GESIS study id, e.g. ZA10000")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output JSON file path. Defaults to <study_id>_variables.json in the current directory.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Number of records requested per API call.",
    )
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Embed the complete `_source` payload for each variable under the `_raw` key.",
    )
    parser.add_argument(
        "--short-output",
        "-s",
        type=Path,
        help="Optional path for a compact mapping JSON: {variable_name: question}. If not provided, a sibling '<name>_short.json' will be written next to --output (or the default output path).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON with indentation instead of the compact representation.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    study_id = args.study_id.strip()
    if not study_id:
        raise SystemExit("study_id cannot be empty")

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")
    logging.info("Fetching variables for study %s", study_id)

    raw_records = fetch_variables_for_study(study_id, page_size=args.page_size)
    if not raw_records:
        logging.warning("No variables found for study %s", study_id)

    payload = build_payload(study_id, raw_records,
                            include_raw=args.include_raw)

    output_path = args.output or Path.cwd() / f"{study_id}_variables.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        if args.pretty:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        else:
            json.dump(payload, fh, ensure_ascii=False)

    # Prepare and write compact mapping JSON
    def _derive_short_path(base: Path) -> Path:
        stem = base.stem
        suffix = base.suffix or ".json"
        return base.with_name(f"{stem}_short{suffix}")

    short_output_path: Path = args.short_output or _derive_short_path(
        output_path)
    short_map = {
        var_name: (rec.get("question_en") or rec.get("title_en")
                   or rec.get("question_de") or rec.get("title_de") or "")
        for var_name, rec in payload["variables"].items()
    }
    with short_output_path.open("w", encoding="utf-8") as fh:
        if args.pretty:
            json.dump(short_map, fh, ensure_ascii=False, indent=2)
        else:
            json.dump(short_map, fh, ensure_ascii=False)
    logging.info("Wrote compact mapping to %s", short_output_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
