

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import pandas as pd

    from data_loader import load_ufo_data

    data = load_ufo_data()
    data.dropna(subset=["comments"], inplace=True)

    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.probability import FreqDist
    from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures

    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("averaged_perceptron_tagger_eng")
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    bigram_measures = BigramAssocMeasures()


@app.cell
def _():
    data
    return


@app.class_definition
class LanguageProcessor:
    def __init__(self, data):
        # Store the data as an instance variable
        self.data = data
        # Process tokens after initializing the data
        self._process_tokens()

    def _process_tokens(self):
        # First create the tokens column
        self.data["tokens"] = self.data["comments"].apply(self._preprocess)
        # Then initialize all_tokens and fdist
        self.all_tokens = [t for tokens in self.data["tokens"] for t in tokens]
        self.fdist = FreqDist(self.all_tokens)

    def _preprocess(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"/", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return [lemmatizer.lemmatize(t) for t in tokens]

    def get_top_words(self, n=20) -> pd.DataFrame:
        words_df = pd.DataFrame(self.fdist.items(), columns=["word", "count"])
        words_df = (
            words_df.sort_values("count", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        return words_df

    def get_processed_data(self) -> pd.DataFrame:
        # Return the processed dataframe
        return self.data

    def get_bigrams(self, n=10):
        finder = BigramCollocationFinder.from_words(self.all_tokens)
        return finder.nbest(bigram_measures.pmi, n)

    def get_pos_tags(self):
        sample = data.loc[0, "tokens"]
        return nltk.pos_tag(sample)


@app.cell
def _():
    processor = LanguageProcessor(data)
    processor.get_processed_data()
    return (processor,)


@app.cell
def _(processor):
    print("Top 10 bigrams:", processor.get_bigrams())
    # this feels like a really useless result
    return


@app.cell
def _(processor):
    print("POS tags:", processor.get_pos_tags())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        TODO:

        - TF-IDF & Clustering with sklearn to group similar comments together?
        - Topic modeling?
        - Sentiment Analysis?
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
