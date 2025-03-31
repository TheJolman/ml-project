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
                          "latitude": str,
                      })
    data.rename(columns={data.columns[10]: "longitude"}, inplace=True)
    data["duration (seconds)"] = data["duration (seconds)"].replace(
        to_replace=r'[^0-9\.]',
        value='',
        regex=True
    ).astype(float)
    data["latitude"] = data["latitude"].replace("33q.200088", "33.200088")
    data["latitude"] = data["latitude"].astype(float)
    data.columns
    return data, pd


@app.cell
def _(data):
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    from cartopy import crs as ccrs, feature as cfeature
    warnings.filterwarnings('ignore')

    fig = plt.figure(figsize=(11, 8.5))
    ax = plt.subplot(1, 1, 1,
                     projection=ccrs.PlateCarree(central_longitude=-75))
    ax.set_title("UFO sightings heat map")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='red')
    ax.set_facecolor("white")
    fig

    lon, lat = np.mgrid[-180:181, -90:91]

    plt.scatter(data['longitude'], data['latitude'],
               transform=ccrs.PlateCarree(),
               alpha=0.5, c="red", s=5)
    return ax, ccrs, cfeature, fig, lat, lon, np, plt, warnings


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
