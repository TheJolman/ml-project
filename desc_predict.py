

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

        This model attempts to predict the likely desciptors of a UFO sighting based off geographic location.

        1. Geographic coordinates are first encoded as geohash strings, which are fed into an ebedding layer in order to create a dense representation of location.
            - Geohashes are transformed into 32-dimensional vectors that the nueral net can work with.

        2. Location features are then passed through a neural net that outputs a text embedding vector.
            - Vector represents likely descriptors for a UFO sighting at that location.
        """
    )
    return


@app.function
def get_lang_splits():
    processor = LanguageProcessor()
    data = processor.get_processed_data()
    data.dropna(subset=["latitude", "longitude", "tokens"], inplace=True)
    data = data[data["tokens"].apply(len) > 0]

    train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    return train_df, val_df, test_df


@app.cell
def _():
    train_df, val_df, test_df = get_lang_splits()

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


@app.cell(disabled=True)
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


@app.class_definition
# Exportable class that does the same as above
class UFODescriptorPredictor:
    """
    A model that predicts likely UFO sighting descriptors based on geographic location.

    This model converts geographic coordinates to geohash strings, embeds them,
    and uses a neural network to predict text embeddings that represent
    likely descriptors for UFO sightings at the given location.
    """

    def __init__(
        self,
        model_path=None,
        geohash_precision=7,
        text_model_name="all-MiniLM-L6-v2",
    ):
        """
        Initialize the UFO descriptor predictor.

        Args:
            model_path: Path to a saved model. If None, a new model will be created.
            geohash_precision: Precision level for geohash encoding.
            text_model_name: Name of the SentenceTransformer model to use.
        """
        self.geohash_precision = geohash_precision
        self.text_model = SentenceTransformer(text_model_name)
        self.text_embedding_dim = (
            self.text_model.get_sentence_embedding_dimension()
        )

        # Will be set during training or loading
        self.model_path = model_path
        self.model = None
        self.geohash_to_idx = None
        self.vocab_size = None

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def is_model_cached(self) -> bool:
        """
        Returns true if a model is cached on disk.
        """
        if self.model_path and os.path.exists(self.model_path):
            return True

    def _create_model(
        self,
        vocab_size,
        coord_embedding_dim=32,
        hidden_dim=128,
        learning_rate=1e-4,
    ):
        """
        Create the neural network model architecture.

        Args:
            vocab_size: Size of the geohash vocabulary.
            coord_embedding_dim: Dimension of the coordinate embedding.
            hidden_dim: Dimension of the hidden layer.
            learning_rate: Learning rate for the Adam optimizer.

        Returns:
            A compiled Keras model.
        """
        input_coord = keras.Input(
            shape=(1,), dtype="int32", name="geohash_index"
        )

        coord_embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=coord_embedding_dim,
            name="coordinate_embedding",
        )(input_coord)

        coord_flat = keras.layers.Flatten()(coord_embedding)

        hidden_layer = keras.layers.Dense(
            hidden_dim, activation="relu", name="hidden_layer"
        )(coord_flat)

        dropout_layer = keras.layers.Dropout(0.2)(hidden_layer)

        output_layer = keras.layers.Dense(
            self.text_embedding_dim,
            activation="linear",
            name="predicted_text_embedding",
        )(dropout_layer)

        model = keras.Model(
            inputs=input_coord, outputs=output_layer, name="CoordToTextMapper"
        )

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        cosine_loss = keras.losses.CosineSimilarity(axis=1)

        model.compile(
            optimizer=optimizer,
            loss=cosine_loss,
            metrics=[keras.metrics.CosineSimilarity(name="cosine_similarity")],
        )

        return model

    def _coords_to_geohash(self, lat, lon):
        """
        Convert latitude and longitude to a geohash string.

        Args:
            lat: Latitude value.
            lon: Longitude value.

        Returns:
            A geohash string.
        """
        return pgh.encode(lat, lon, precision=self.geohash_precision)

    def _prepare_data(self, df):
        """
        Prepare data for training by adding geohash column.

        Args:
            df: DataFrame with 'latitude' and 'longitude' columns.

        Returns:
            DataFrame with added 'geohash' column.
        """
        df = df.copy()
        df["geohash"] = df.apply(
            lambda row: self._coords_to_geohash(
                row["latitude"], row["longitude"]
            ),
            axis=1,
        )
        return df

    def _build_vocab(self, geohashes):
        """
        Build vocabulary mapping from geohash strings to indices.

        Args:
            geohashes: Series or list of geohash strings.
        """
        unique_geohashes = np.unique(geohashes)
        self.geohash_to_idx = {gh: i for i, gh in enumerate(unique_geohashes)}
        self.vocab_size = len(unique_geohashes)
        print(f"Vocabulary built with {self.vocab_size} unique geohashes.")

    def _geohash_to_index(self, geohash):
        """
        Convert a geohash string to its vocabulary index.

        Args:
            geohash: A geohash string.

        Returns:
            Integer index of the geohash in the vocabulary.
            Returns 0 for unknown geohashes (handles unseen locations).
        """
        # Default to index 0 for unknown geohashes
        if self.geohash_to_idx is None:
            raise ValueError(
                "Vocabulary not initialized. Call build_vocab first."
            )
        return self.geohash_to_idx.get(geohash, 0)

    def _encode_text(self, descriptions):
        """
        Encode text descriptions into embeddings.

        Args:
            descriptions: List of text descriptions or tokens.

        Returns:
            Numpy array of text embeddings.
        """
        if descriptions.empty:
            print("Warning: attempting to encode empty descriptions Series.")
            return np.empty((0, self.text_embedding_dim), dtype=np.float32)

        first_element = descriptions.iloc[0]

        # Handle both string descriptions and token lists
        if isinstance(first_element, list):
            sentences = [" ".join(map(str, tokens)) for tokens in descriptions]
        else:
            sentences = descriptions.tolist()

        print(f"Encoding {len(sentences)} descriptions...")
        embeddings = self.text_model.encode(sentences, show_progress_bar=True)
        return embeddings.astype(np.float32)

    def train(
        self, train_df, val_df=None, epochs=20, batch_size=64, save_path=None
    ):
        """
        Train the model on the provided data.

        Args:
            train_df: Training DataFrame with 'latitude', 'longitude', and 'tokens' columns.
            val_df: Validation DataFrame with the same columns.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            save_path: Path to save the trained model.

        Returns:
            Training history.
        """
        if train_df.empty:
            raise ValueError("Training DataFrame cannot be empty.")

        train_data = self._prepare_data(train_df)
        self._build_vocab(train_data["geohash"])
        # TODO: Maybe make this a public func and check if not none here
        self.model = self._create_model(self.vocab_size)

        X_train = np.array(
            [self._geohash_to_index(gh) for gh in train_data["geohash"]]
        )

        Y_train = self._encode_text(train_data["tokens"])

        if X_train.shape[0] == 0:
            raise ValueError(
                "Training data processing resulted in empty features (X_train)."
            )
        if Y_train.shape[0] == 0:
            raise ValueError(
                "Training data encoding resulted in empty labels (Y_train)."
            )
        if X_train.shape[0] != Y_train.shape[0]:
            raise ValueError(
                f"Training features and labels have mismatched sample counts: {X_train.shape[0]} != {Y_train.shape[0]}"
            )

        # Prepare validation data if provided
        if val_df is not None:
            val_data = self._prepare_data(val_df)
            X_val = np.array(
                [self._geohash_to_index(gh) for gh in val_data["geohash"]]
            )
            Y_val = self._encode_text(val_data["tokens"])
            validation_data = (X_val, Y_val)
        else:
            validation_data = None

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, restore_best_weights=True
            )
        ]

        print("Model training started...")
        history = self.model.fit(
            X_train,
            Y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
        )

        if save_path:
            self._save_model(save_path)

        return history

    def predict(self: float, lat: float, lon):
        """
        Predict text embedding for a given location.

        Args:
            lat: Latitude value.
            lon: Longitude value.

        Returns:
            Predicted text embedding vector.
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained yet.")

        geohash = self._coords_to_geohash(lat, lon)
        geohash_idx = np.array([self._geohash_to_index(geohash)])

        return self.model.predict(geohash_idx)[0]

    def predict_batch(self, lats, lons):
        """
        Predict text embeddings for multiple locations.

        Args:
            lats: List of latitude values.
            lons: List of longitude values.

        Returns:
            Array of predicted text embedding vectors.
        """
        if self.model is None:
            raise ValueError("Model not loaded or trained yet.")

        geohashes = [
            self._coords_to_geohash(lat, lon) for lat, lon in zip(lats, lons)
        ]
        geohash_indices = np.array(
            [self._geohash_to_index(gh) for gh in geohashes]
        )

        return self.model.predict(geohash_indices)

    def find_similar_descriptions(
        self, embedding, descriptions, top_n=5, precomputed_embeddings=None
    ):
        """
        Find the most similar descriptions to a given embedding.

        Args:
            embedding: Text embedding vector.
            descriptions: List of text descriptions.
            top_n: Number of top results to return.
            precomputed_embeddings: Optional precomputed embeddings for descriptions.
                                   If provided, skips the encoding step.

        Returns:
            List of (description, similarity_score) tuples.
        """
        if precomputed_embeddings is not None:
            description_embeddings = precomputed_embeddings
            if len(description_embeddings) != len(descriptions):
                raise ValueError(
                    f"Number of precomputed embeddings ({len(precomputed_embeddings)}) "
                    f"does not match number of descriptions ({len(descriptions)})"
                )
        else:
            description_embeddings = self.text_model.encode(descriptions)

        # Calculate cosine similarities
        similarities = np.dot(description_embeddings, embedding) / (
            np.linalg.norm(description_embeddings, axis=1)
            * np.linalg.norm(embedding)
        )

        # Get top N results
        top_indices = np.argsort(similarities)[-top_n:][::-1]

        return [(descriptions[i], similarities[i]) for i in top_indices]

    def _save_model(self, path):
        """
        Save the model and vocabulary to disk.

        Args:
            path: Directory path to save the model.
        """
        if self.model is None:
            raise ValueError("No model to save.")

        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model")
        self.model.save(model_path)

        vocab_path = os.path.join(path, "vocab.npz")
        np.savez(
            vocab_path,
            geohash_to_idx=np.array(
                list(self.geohash_to_idx.items()), dtype=object
            ),
            geohash_precision=np.array([self.geohash_precision]),
        )

        print(f"Model and vocabulary saved to {path}")

    def _load_model(self, path):
        """
        Load the model and vocabulary from disk.

        Args:
            path: Directory path containing the saved model.
        """
        model_path = os.path.join(path, "model")
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")

        vocab_path = os.path.join(path, "vocab.npz")
        if os.path.exists(vocab_path):
            vocab_data = np.load(vocab_path, allow_pickle=True)
            self.geohash_to_idx = dict(vocab_data["geohash_to_idx"])
            self.vocab_size = len(self.geohash_to_idx)
            self.geohash_precision = int(vocab_data["geohash_precision"][0])
        else:
            raise FileNotFoundError(f"Vocabulary not found at {vocab_path}")

        # Verify model output dimension matches text embedding dimension
        if self.model.output_shape[1] != self.text_embedding_dim:
            raise ValueError(
                f"Model output dimension ({self.model.output_shape[1]}) "
                f"does not match text embedding dimension ({self.text_embedding_dim})"
            )

        print(f"Model loaded with vocabulary size {self.vocab_size}")


@app.cell
def _(mo):
    mo.md(r"""## Using the model""")
    return


@app.cell
def _():
    pred = UFODescriptorPredictor(model_path="./outputs/text_predictor_model")
    pred.is_model_cached()
    pred.predict(90, 90)
    return


@app.function
def demo_ufo_predictor(train_df, val_df, use_existing_model=True):
    # Initialize the predictor with existing model or train a new one
    if use_existing_model and os.path.exists("./outputs/text_predictor_model"):
        predictor = UFODescriptorPredictor(
            model_path="./outputs/text_predictor_model"
        )
        print("Loaded existing model")
    else:
        print("Training new model...")
        predictor = UFODescriptorPredictor()
        predictor.train(
            train_df, val_df, save_path="./outputs/text_predictor_model"
        )
        print("Model training complete")

    # Make predictions for a location
    lat, lon = 37.7749, -122.4194  # San Francisco
    embedding = predictor.predict(lat, lon)

    # Find similar descriptions from a corpus
    sample_descriptions = [
        "bright light hovering in the sky",
        "triangular craft moving silently",
        "pulsating orb changing colors",
        "disk-shaped object with flashing lights",
        "cigar-shaped object moving at high speed",
    ]

    similar = predictor.find_similar_descriptions(
        embedding, sample_descriptions
    )

    print(f"Predicted UFO descriptions for location ({lat}, {lon}):")
    for desc, score in similar:
        print(f"- {desc} (similarity: {score:.4f})")

    return predictor


@app.cell
def _():
    te, va, tr = get_lang_splits()
    demo_ufo_predictor(te, va, use_existing_model=True)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
