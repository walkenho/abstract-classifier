"""Load data functionalities."""
from __future__ import annotations

import pathlib
import pickle
from typing import Dict, Tuple

import numpy as np


def load_processed_data(
    folder: pathlib.Path,
) -> Tuple[Tuple[np.array, np.array, np.array, np.array, np.array, np.array], np.array]:
    """Load train, validation and test data and labels from folder."""
    x_train = np.load(folder / "X_train.npy", allow_pickle=True)
    x_val = np.load(folder / "X_val.npy", allow_pickle=True)
    x_test = np.load(folder / "X_test.npy", allow_pickle=True)

    y_train = np.load(folder / "y_train.npy", allow_pickle=True)
    y_val = np.load(folder / "y_val.npy", allow_pickle=True)
    y_test = np.load(folder / "y_test.npy", allow_pickle=True)

    labels = np.load(folder / "labels.npy", allow_pickle=True)

    return (x_train, x_val, x_test, y_train, y_val, y_test), labels


def load_taxonomy(filepath: pathlib.Path) -> Dict:
    """Load arXiv taxonomy from filepath."""
    with open(filepath, "rb") as f:
        return pickle.load(f)
