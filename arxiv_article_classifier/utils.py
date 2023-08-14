"""General Utility Functions."""
from __future__ import annotations

import pandas as pd
from IPython.display import display


def display_fully(df: pd.DataFrame) -> None:
    """Display pandas dataframe in full."""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 2000)
    pd.set_option("display.float_format", "{:20,.2f}".format)
    pd.set_option("display.max_colwidth", None)
    display(df)
    pd.reset_option("display.max_rows")
    pd.reset_option("display.max_columns")
    pd.reset_option("display.width")
    pd.reset_option("display.float_format")
    pd.reset_option("display.max_colwidth")


def save_image(fig, path, filename) -> None:
    """Save image to png and html."""
    fig.write_image(
        path / f"{filename}.png",
        width=1600,
        height=800,
        scale=2,
    )
    fig.write_html(path / f"{filename}.html")
