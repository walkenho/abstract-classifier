"""Test data related functions."""
from __future__ import annotations

from arxiv_article_classifier.data.make_processed_data_tfidf import (
    LATEX_REGEX,
    delete_regular_expression,
)


def test_delete_regular_expression_removes_latex():
    """Assert that delete_regular_expression deletes a latex expression."""
    assert delete_regular_expression("Some $latex-formatted$ text", LATEX_REGEX) == "Some   text"
