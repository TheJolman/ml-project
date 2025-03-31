import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("sahityasetu/ufo-sightings")

    print("Path to dataset files:", path)
    return kagglehub, mo, path


@app.cell
def _(path):
    import pandas as pd
    data = pd.read_csv(f"{path}/ufo_sightings_scrubbed.csv")
    data

    return data, pd


if __name__ == "__main__":
    app.run()
