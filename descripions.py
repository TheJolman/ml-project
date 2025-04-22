

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    from data_loader import load_ufo_data

    data = load_ufo_data()
    data.head()
    return (data,)


@app.cell
def _():
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
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    bigram_measures = BigramAssocMeasures()
    return lemmatizer, re, stop_words, word_tokenize


@app.cell
def _(data, lemmatizer, re, stop_words, word_tokenize):
    def preprocess(text: str) -> list[str]:
        text = re.sub(r"[^a-zA-Z\s]", "", text.lower())
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return [lemmatizer.lemmatize(t) for t in tokens]


    data["tokens"] = data["comments"].apply(preprocess)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
