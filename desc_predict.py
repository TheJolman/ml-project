

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import numpy as np
    import pandas as pd

    import pygeohash as pgh
    from sklearn.model_selection import train_test_split
    from sklearn.metrics.pairwise import cosine_similarity

    from sentence_transformers import SentenceTransformer
    import tensorflow as tf
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"
    try:
        from tensorflow import keras
    except ImportError as e:
        print(f"Error importing Keras: {e}")

    from descriptions import LanguageProcessor

    MODEL_PATH = "./outputs/text_predictor_model"


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
    return test, train, val


@app.cell
def _(train):
    geohash_vocab = train["geohash"].unique()
    geohash_to_idx = {gh: i for i, gh in enumerate(geohash_vocab)}
    vocab_size = len(geohash_vocab)

    print(f"There are {vocab_size} unique geohash strings.")
    return geohash_to_idx, vocab_size


@app.cell
def _(geohash_to_idx, test, train, val):
    coord_embedding_dim = 32


    def geohashes_to_indices(df, mapping):
        try:
            df["geohash"]
        except KeyError:
            print("Column 'geohash' doesn't exist.")
        return df["geohash"].apply(lambda gh: mapping.get(gh, 0)).values


    X_train_coords = geohashes_to_indices(train, geohash_to_idx)
    X_val_coords = geohashes_to_indices(val, geohash_to_idx)
    X_test_coords = geohashes_to_indices(test, geohash_to_idx)
    return X_test_coords, X_train_coords, X_val_coords, coord_embedding_dim


@app.cell
def _(mo):
    mo.md(r"""## Text encoding""")
    return


@app.cell
def _():
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    return (text_model,)


@app.cell
def _(text_model):
    text_embedding_dim = text_model.get_sentence_embedding_dimension()
    print("Text embedding dimension:", text_embedding_dim)
    return (text_embedding_dim,)


@app.cell
def _(mo, test, text_model, train, val):
    # this is an expensive function call
    @mo.persistent_cache
    def get_text_embeddings(df, model):
        """Encodes 'tokens' col of a DataFrame into sentence embeddings."""
        sentences = (
            df["tokens"].apply(lambda tokens: " ".join(map(str, tokens))).tolist()
        )

        print(f"Encoding {len(sentences)} descriptions...")
        embeddings = model.encode(sentences, show_progress_bar=True)
        return embeddings.astype(np.float32)


    Y_train_text = get_text_embeddings(train, text_model)
    Y_val_text = get_text_embeddings(val, text_model)
    Y_test_text = get_text_embeddings(test, text_model)

    print("Shape of Y_train_text:", Y_train_text.shape)
    print("Shape of Y_val_text:", Y_val_text.shape)
    print("Shape of Y_test_text:", Y_test_text.shape)
    return Y_test_text, Y_train_text, Y_val_text


@app.cell
def _(mo):
    mo.md(r"""## Training model with Keras""")
    return


@app.cell
def _(text_embedding_dim):
    HIDDEN_DIM = 128
    OUTPUT_DIM = text_embedding_dim
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    BATCH_SIZE = 64

    input_coord = keras.Input(shape=(1,), dtype="int32", name="geohash_index")
    return (
        BATCH_SIZE,
        EPOCHS,
        HIDDEN_DIM,
        LEARNING_RATE,
        OUTPUT_DIM,
        input_coord,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Embedding layer for Geohashes
        **Input dimension:** vocab size  
        **Output dimension:** coord embedding size
        """
    )
    return


@app.cell
def _(HIDDEN_DIM, OUTPUT_DIM, coord_embedding_dim, input_coord, vocab_size):
    coord_embedding = keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=coord_embedding_dim,
        name="coordinate_embedding",
    )(input_coord)

    coord_flat = keras.layers.Flatten()(coord_embedding)

    hidden_layer = keras.layers.Dense(
        HIDDEN_DIM, activation="relu", name="hidden_layer"
    )(coord_flat)

    dropout_layer = keras.layers.Dropout(0.2)(hidden_layer)

    output_layer = keras.layers.Dense(
        OUTPUT_DIM, activation="linear", name="predicted_text_embedding"
    )(dropout_layer)

    model = keras.Model(
        inputs=input_coord, outputs=output_layer, name="CoordToTextMapper"
    )

    model.summary()
    return (model,)


@app.cell
def _(LEARNING_RATE, model):
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    cosine_loss = keras.losses.CosineSimilarity(axis=1)

    model.compile(
        optimizer=optimizer,
        loss=cosine_loss,
        metrics=[keras.metrics.CosineSimilarity(name="cosine_similarity")],
    )
    return


@app.cell
def _(
    BATCH_SIZE,
    EPOCHS,
    X_train_coords,
    X_val_coords,
    Y_train_text,
    Y_val_text,
    mo,
    model,
    path,
):
    @mo.cache
    def train_model():
        print("Model training started...")

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )

        history = model.fit(
            X_train_coords,
            Y_train_text,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val_coords, Y_val_text),
            callbacks=[early_stopping],
        )

        print("Training done!")
        print(f"Saving model to {MODEL_PATH}...")
        try:
            model.save(path)
            print("Model saved.")
        except Exception as e:
            print("Error saving model:", e)
        return model
    return (train_model,)


@app.cell
def _(train_model):
    def get_trained_model(train_new=False):
        if os.path.exists(MODEL_PATH) and not train_new:
            return keras.models.load_model(MODEL_PATH)
        return train_model()


    trained_model = get_trained_model()
    return (trained_model,)


@app.cell
def _(BATCH_SIZE, X_test_coords, Y_test_text, trained_model):
    test_loss, test_cosine_similarity = trained_model.evaluate(
        X_test_coords, Y_test_text, batch_size=BATCH_SIZE
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Cosine Similarity: {test_cosine_similarity:.4f}")
    return


@app.cell
def _(mo):
    mo.md(r"""## Using the model""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
