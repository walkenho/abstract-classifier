"""Prepare data sets for ML."""
from __future__ import annotations

import ast
import logging
import pathlib
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split

from arxiv_article_classifier.data.scrape_arxiv import CATEGORIES_OF_INTEREST

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("make_interim_data.py")


def multilabel_stratefied_train_validation_test_split(
    X: np.array,
    y: np.array,
    test_size: float = None,
    validation_size: float = None,
) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """Split arrays into train, validation and test using iterative multilabel stratification."""
    # TODO: If too slow for larger datasets, replace by
    # https://datascience.stackexchange.com/questions/45174/how-to-use-sklearn-train-test-split-to-stratify-data-for-multi-label-classificat
    x_train, y_train, x_val_test, y_val_test = iterative_train_test_split(
        X,
        y,
        test_size=test_size + validation_size,
    )

    x_val, y_val, x_test, y_test = iterative_train_test_split(
        x_val_test,
        y_val_test,
        test_size=test_size / (test_size + validation_size),
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def make_interim_data(
    input_file: pathlib.Path,
    output_folder,
    categories_to_keep,
    test_size: float = 0.2,
    validation_size: float = 0.2,
) -> None:
    """Make interim dataset and save it to disk.

    Loads abstract data from input_folder/input_file, and performs a multilabel, stratified split.
    """
    logger.info("Entering make_interim_data.")
    logger.info("Read data.")
    df = pd.read_csv(input_file).assign(
        tags=lambda df: df["tags"].apply(
            lambda x: [tag for tag in ast.literal_eval(x) if tag in categories_to_keep]
        )
    )

    abstracts = df["abstracts"].values.reshape(-1, 1)

    mlb = MultiLabelBinarizer()
    tag_matrix = mlb.fit_transform(df["tags"])

    (
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
    ) = multilabel_stratefied_train_validation_test_split(
        abstracts, tag_matrix, test_size=test_size, validation_size=validation_size
    )

    logger.info("Save data to %s.", output_folder)
    np.save(output_folder / "X_train.npy", x_train.flatten())
    np.save(output_folder / "X_val.npy", x_val.flatten())
    np.save(output_folder / "X_test.npy", x_test.flatten())
    np.save(output_folder / "y_train.npy", y_train)
    np.save(output_folder / "y_val.npy", y_val)
    np.save(output_folder / "y_test.npy", y_test)
    np.save(output_folder / "labels.npy", mlb.classes_)


if __name__ == "__main__":
    datafolder = pathlib.Path(__file__).parent.parent.parent / "data"
    make_interim_data(
        input_file=datafolder / "raw" / "articles.csv",
        output_folder=datafolder / "interim",
        categories_to_keep=CATEGORIES_OF_INTEREST,
        test_size=0.2,
        validation_size=0.2,
    )
