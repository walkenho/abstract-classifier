"""Create a baseline model for a multilabel classification task."""
from __future__ import annotations

from typing import Dict, List, Union

import numpy as np


class DictionaryModel:
    """Baseline Model Class."""

    def __init__(self, keywords: Dict, labelorder: Union[List, np.array]):
        """Create DictionaryModel instance."""
        self.keywords = keywords
        self.labelorder = labelorder

    def predict(self, abstracts: Union[np.array, List]) -> np.array:
        """Predict labels for input texts and return an array.

        Returns True for label/message combinations
        where associated keywords can be found in message.
        """
        # Use labels to ensure arrays are returned in correct order
        return np.array(
            [
                [
                    len(set(text.split()).intersection(self.keywords[label])) > 0
                    for label in self.labelorder
                ]
                for text in abstracts
            ]
        )
