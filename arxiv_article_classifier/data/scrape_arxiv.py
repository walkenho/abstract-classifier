"""Code to extract data from arXiv."""
from __future__ import annotations

import logging
import pickle
import sys
from collections import namedtuple
from pathlib import Path
from typing import Dict, List

import arxiv
import pandas as pd
import requests
from bs4 import BeautifulSoup
from pandas.core.base import PandasObject
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("scrape_arxiv.py")

SearchResults = namedtuple("SearchResults", ["ids", "titles", "abstracts", "tags"])
Taxonomy = namedtuple("Taxonomy", ["abbreviation", "description"])

CATEGORIES_OF_INTEREST = [
    "cs.AI",
    "cs.CL",
    "cs.CV",
    "cs.CY",
    "cs.GT",
    "cs.LG",
    "cs.MA",
    "cs.RO",
    "cs.SI",
    "eess.AS",
    "eess.IV",
    "eess.SP",
    "eess.SY",
    "math.OC",
    "math.ST",
    "math.NA",
]


def download_taxonomy() -> Dict:
    """Download arXiv taxonomy and return a dictionary."""

    def _cleanly_split_taxonomy_line(line) -> Taxonomy:
        """Split taxonomy line into abbreviation and description."""
        abbreviation, description = line.split("(")
        return Taxonomy(abbreviation.strip(), description.strip(")").strip())

    logger.info("Downloading arXiv taxonomy")
    response = requests.get("https://arxiv.org/category_taxonomy", timeout=10)
    if response.status_code != 200:
        logger.error("Could not reach https://arxiv.org/category_taxonomy")
        raise ConnectionRefusedError("Could not reach https://arxiv.org/category_taxonomy")
    soup = BeautifulSoup(response.text, features="xml")
    h4s = soup.find_all("h4")
    logger.info("Found %s entries.", len(h4s))

    return {
        entry.abbreviation: entry.description
        for entry in [_cleanly_split_taxonomy_line(h4.text) for h4 in h4s[1:]]
    }


def find_articles_by_category(
    client, categories: List, max_results=1, sort_by=arxiv.SortCriterion.SubmittedDate
) -> SearchResults:
    """Find arxiv article details and return SearchResults.

    client: arxiv-client
    categories: categories to search
    max_results: number of results
    sort_by: arxiv sort criterium

    returns:
    SearchResults
    """
    titles = []
    abstracts = []
    tags = []
    ids = []

    logger.info("Downloading arXiv data set")
    for category in categories:
        logger.info("Searching articles for category %s", category)
        search = arxiv.Search(query=f"cat:{category}", max_results=max_results, sort_by=sort_by)
        for result in tqdm(client.results(search), desc=category):
            tags.append(result.categories)
            titles.append(result.title)
            abstracts.append(result.summary)
            ids.append(result.entry_id)
        logger.info("Downloaded %s articles.", len(ids))

    return SearchResults(ids, titles, abstracts, tags)


def main():
    """Download arXiv taxonomy and article abstract."""

    def _deduplicate_by_id(data: pd.DataFrame):
        return data.groupby("ids").first()

    PandasObject.deduplicate_by_id = _deduplicate_by_id

    datapath = Path(__file__).parent.parent.parent / "data" / "raw"
    arxiv_client = arxiv.Client(num_retries=20, page_size=500, delay_seconds=3)
    results = find_articles_by_category(arxiv_client, CATEGORIES_OF_INTEREST, max_results=1000)

    data = pd.DataFrame(dict(zip(results._fields, results))).deduplicate_by_id()
    logger.info("Deduplicated data. %s articles left.", data.shape[0])
    filepath_arxiv_data = datapath / "articles.csv"
    data.to_csv(filepath_arxiv_data)
    logger.info("Saved arXiv data to %s", filepath_arxiv_data)

    taxonomy = download_taxonomy()
    filepath_taxonomy = datapath / "taxonomy.pkl"
    with open(filepath_taxonomy, "wb") as f:
        pickle.dump(taxonomy, f)
    logger.info("Saved taxonomy to %s", filepath_taxonomy)


if __name__ == "__main__":
    main()
