

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import numpy as np
    import pandas as pd

    from descriptions import LanguageProcessor


@app.cell
def _():
    processor = LanguageProcessor()
    processor.get_top_words()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
