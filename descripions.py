

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd

    from data_loader import load_ufo_data

    data = load_ufo_data()
    data.dropna(subset=["comments"], inplace=True)
    data
    return data, pd


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
    nltk.download("averaged_perceptron_tagger_eng")
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    bigram_measures = BigramAssocMeasures()
    return (
        BigramCollocationFinder,
        FreqDist,
        bigram_measures,
        lemmatizer,
        nltk,
        re,
        stop_words,
        word_tokenize,
    )


@app.cell
def _(data, lemmatizer, re, stop_words, word_tokenize):
    def preprocess(text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"/", " ", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return [lemmatizer.lemmatize(t) for t in tokens]


    data["tokens"] = data["comments"].apply(preprocess)
    data["tokens"].head()
    return


@app.cell
def _(FreqDist, data):
    all_tokens = [t for tokens in data["tokens"] for t in tokens]
    fdist = FreqDist(all_tokens)
    return all_tokens, fdist


@app.cell
def _(fdist, pd):
    words_df = pd.DataFrame(fdist.items(), columns=["word", "count"])
    words_df = words_df.sort_values("count", ascending=False).reset_index(
        drop=True
    )
    words_df
    return


@app.cell
def _(BigramCollocationFinder, all_tokens, bigram_measures):
    finder = BigramCollocationFinder.from_words(all_tokens)
    top_bigrams = finder.nbest(bigram_measures.pmi, 10)
    print("Top 10 bigrams:", top_bigrams)
    # this feels like a really useless result
    return


@app.cell
def _(data, nltk):
    sample = data.loc[0, "tokens"]
    print("POS tags:", nltk.pos_tag(sample))
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        TODO:

        - TF-IDF & Clustering with sklearn to group similar comments together
        - Topic modeling?
        - Sentiment Analysis
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
