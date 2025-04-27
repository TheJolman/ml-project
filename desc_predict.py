

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import numpy as np
    import pandas as pd
    import pygeohash as pgh
    from sklearn.model_selection import train_test_split

    from descriptions import LanguageProcessor


@app.cell
def _(mo):
    mo.md(
        r"""
        # Model: Coords in, likely descriptors out

        This model attempts to predict the likely desciptors of a UFO sighting based off the geographic location it occured at.
        """
    )
    return


@app.cell
def _():
    processor = LanguageProcessor()
    data = processor.get_processed_data()
    data.dropna(subset=["latitude", "longitude", "tokens"], inplace=True)
    data = data[data["tokens"].apply(len) > 0]
    data
    return (data,)


@app.cell
def _(data):
    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print("Training samples:", len(train_df))
    print("Validation samples:", len(val_df))
    print("Testing samples:", len(test_df))
    return test_df, train_df, val_df


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Coordinate Encoding
        Use geospacial tiling and an embedding layer for coordinate encoding:

        - convert (lat, lon) to geohash strings & use as categorical features
        - learn embedding vector to for each unique tile ID
        """
    )
    return


@app.cell
def _(test_df, train_df, val_df):
    GEOHASH_PRECISION = 7


    def coords_to_geohash(df, precision) -> pd.DataFrame:
        df["geohash"] = df.apply(
            lambda row: pgh.encode(
                row["latitude"], row["longitude"], precision=precision
            ),
            axis=1,
        )
        return df


    train = coords_to_geohash(train_df.copy(), GEOHASH_PRECISION)
    val = coords_to_geohash(val_df.copy(), GEOHASH_PRECISION)
    test = coords_to_geohash(test_df.copy(), GEOHASH_PRECISION)
    train[["city", "latitude", "longitude", "geohash"]]
    return train, val


@app.cell
def _(train):
    geohash_vocab = train["geohash"].unique()
    geohash_to_idx = {gh: i for i, gh in enumerate(geohash_vocab)}
    vocab_size = len(geohash_vocab)

    print(f"There are {vocab_size} unique geohash strings.")
    return (geohash_to_idx,)


@app.cell
def _(geohash_to_idx, train, val):
    coord_embedding_dim = 32


    def geohashes_to_indices(df, mapping):
        try:
            df["geohash"]
        except KeyError:
            print("Column 'geohash' doesn't exist.")
        return df["geohash"].apply(lambda gh: mapping.get(gh, 0)).values


    X_train_coords = geohashes_to_indices(train, geohash_to_idx)
    X_val_coords = geohashes_to_indices(val, geohash_to_idx)
    return


@app.cell
def _(mo):
    mo.md(r"""### Text encoding""")
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
