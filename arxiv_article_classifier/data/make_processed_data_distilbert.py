"""Process data for use in DistilBert model."""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import FunctionTransformer, Pipeline

from arxiv_article_classifier.data.make_processed_data_utils import (
    convert_interim_to_processed_data,
    delete_regular_expression,
)

LINEBREAK_REGEX = re.compile("(\n)+")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def make_data_cleaning_pipeline():
    """Create and return data transformation pipeline."""
    linebreak_cleaner = FunctionTransformer(
        lambda X: np.array([delete_regular_expression(x, LINEBREAK_REGEX) for x in X])
    )

    return Pipeline(
        [
            ("clean_linebreaks", linebreak_cleaner),
        ]
    )


if __name__ == "__main__":
    datafolder = Path(__file__).parent.parent.parent / "data"

    output_folder = datafolder / "processed" / "distilbert-model"
    output_folder.mkdir(parents=True, exist_ok=True)

    logger.info("Converting interim to processed data for DistilBert.")
    convert_interim_to_processed_data(
        input_folder=datafolder / "interim",
        output_folder=output_folder,
        pipe=make_data_cleaning_pipeline(),
    )
