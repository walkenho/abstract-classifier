"""Process data for use in tfidf model."""
from __future__ import annotations

import logging
import re
import string
import sys
from pathlib import Path
from typing import List

import numpy as np
import spacy
from nltk.corpus import stopwords
from sklearn.pipeline import FunctionTransformer, Pipeline

from arxiv_article_classifier.data.make_processed_data_utils import (
    convert_interim_to_processed_data,
    delete_regular_expression,
)

STOPLIST = stopwords.words("english") + ["-"]
LINEBREAK_REGEX = re.compile("(\n)+")
LATEX_REGEX = re.compile(r"\$\S+\$")

PUNCTUATION_DELETION_TABLE = str.maketrans("", "", string.punctuation.replace("-", ""))

NLP = spacy.load("en_core_web_sm")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def remove_stopwords(raw_text: str, stoplist: List[str]) -> str:
    """Remove stopwords from input string and return cleaned string."""
    return " ".join([word for word in raw_text.split() if word not in stoplist])


def lemmatize_document(text: str, spacy_model) -> str:
    """Lemmatize input using spacy model.

    This lemmatizer uses POS-tags to help lemmatize
    """
    return " ".join([token.lemma_ for token in spacy_model(str(text))])


def make_data_cleaning_pipeline():
    """Create and return data transformation pipeline."""
    linebreak_cleaner = FunctionTransformer(
        lambda X: np.array([delete_regular_expression(x, LINEBREAK_REGEX) for x in X])
    )

    lower_case_converter = FunctionTransformer(lambda X: np.array([x.lower() for x in X]))
    whitespace_deleter = FunctionTransformer(lambda X: np.array([" ".join(x.split()) for x in X]))

    lemmatizer = FunctionTransformer(lambda X: np.array([lemmatize_document(x, NLP) for x in X]))

    punctuation_deleter = FunctionTransformer(
        lambda X: np.array([x.translate(PUNCTUATION_DELETION_TABLE) for x in X])
    )

    stopword_remover = FunctionTransformer(
        lambda X: np.array([remove_stopwords(x, STOPLIST) for x in X])
    )

    latex_remover = FunctionTransformer(
        lambda X: np.array([delete_regular_expression(x, LATEX_REGEX) for x in X])
    )

    return Pipeline(
        [
            ("clean_linebreaks", linebreak_cleaner),
            ("remove_latex", latex_remover),
            ("lemmatize", lemmatizer),
            ("convert_to_lowercase", lower_case_converter),
            ("delete_punctuation", punctuation_deleter),
            ("delete_whitespace", whitespace_deleter),
            ("remove_stopwords", stopword_remover),
        ]
    )


if __name__ == "__main__":
    datafolder = Path(__file__).parent.parent.parent / "data"

    output_folder = datafolder / "processed" / "tfidf-model"
    output_folder.mkdir(parents=True, exist_ok=True)

    convert_interim_to_processed_data(
        input_folder=datafolder / "interim",
        output_folder=output_folder,
        pipe=make_data_cleaning_pipeline(),
    )
    logger.info("Successfully converted interim to processed data for tfidf model.")
