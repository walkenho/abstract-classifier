"""Data cleaning functionality."""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import numpy as np

from arxiv_article_classifier.data.load import load_processed_data

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def delete_regular_expression(raw_text: str, regex) -> str:
    """Replace regular expression with empty space and return."""
    return re.sub(regex, " ", raw_text)


def convert_interim_to_processed_data(input_folder: Path, output_folder: Path, pipe) -> None:
    """Clean interim data and store to disk."""
    logger.debug("Inside convert_interim_to_processed_data.")

    (X_train, X_val, X_test, y_train, y_val, y_test), labels = load_processed_data(input_folder)

    logger.info("Transform training data.")
    X_train_cleaned = pipe.transform(X_train)

    logger.info("Transform validation data.")
    X_val_cleaned = pipe.transform(X_val)

    logger.info("Transform test data.")
    X_test_cleaned = pipe.transform(X_test)

    np.save(output_folder / "X_train.npy", X_train_cleaned)
    np.save(output_folder / "X_val.npy", X_val_cleaned)
    np.save(output_folder / "X_test.npy", X_test_cleaned)

    np.save(output_folder / "y_train.npy", y_train)
    np.save(output_folder / "y_val.npy", y_val)
    np.save(output_folder / "y_test.npy", y_test)

    np.save(output_folder / "labels.npy", labels)

    logger.info("Successfully saved data to %s.", output_folder)
