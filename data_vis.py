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
    data = pd.read_csv(f"{path}/ufo_sightings_scrubbed.csv",
                      dtype = {
                          "duration (seconds)": str,
                          "latitude": str
                      })
    data["duration (seconds)"] = data["duration (seconds)"].replace(
        to_replace=r'[^0-9\.]',
        value='',
        regex=True
    ).astype(float)
    data["latitude"] = data["latitude"].replace("33q.200088", "33.200088")
    data["latitude"] = data["latitude"].astype(float)
    data

    return data, pd


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
