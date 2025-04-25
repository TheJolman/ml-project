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
def _():
    import pandas as pd
    from data_loader import load_ufo_data
    data = load_ufo_data()
    data.columns
    return data, load_ufo_data, pd


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
    ax.set_title("UFO sightings scatter plot")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='red')
    ax.set_facecolor("white")

    plt.scatter(data['longitude'], data['latitude'],
                transform=ccrs.PlateCarree(),
                alpha=0.5,
                c="red",
                s=5
                )

    fig.savefig('outputs/ufo_sightings_scatter.png')
    fig
    return ax, ccrs, cfeature, fig, np, plt, warnings


@app.cell
def _(ax, ccrs, cfeature, data, np, plt):
    import matplotlib.colors as colors

    fig2 = plt.figure(figsize=(11, 8.5))
    ax2 = plt.subplot(1, 1, 1,
                     projection=ccrs.PlateCarree(central_longitude=-75))
    ax2.set_title("UFO sightings heatmap")
    ax2.coastlines()
    ax2.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='black')
    ax2.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='red')
    ax2.set_facecolor("white")

    lon, lat = np.mgrid[-180:181, -90:91]

    bins = 200
    heatmap, xedges, yedges = np.histogram2d(
        data['longitude'],
        data['latitude'],
        bins=bins,
        range=[[-180, 180], [-90, 90]]
    )

    heatmap = np.log1p(heatmap)

    lon_bins = np.linspace(-180, 180, bins)
    lat_bins = np.linspace(-90, 90, bins)
    lon_grid, lat_grid = np.meshgrid(lon_bins, lat_bins)

    cs = ax2.pcolormesh(lon_grid, lat_grid, heatmap.T,
                        transform=ccrs.PlateCarree(),
                        cmap='magma',
                        alpha=0.7,
                        norm=colors.PowerNorm(gamma=0.5)
                       )

    cbar = plt.colorbar(cs, ax=ax2,
                        pad=0.1,
                        shrink=0.6
                       )
    cbar.set_label('Log(num sightings)')
    # Why is the colorbar not showing up?

    fig2.savefig('outputs/ufo_sightings_heat.png')
    fig2
    return (
        ax2,
        bins,
        cbar,
        colors,
        cs,
        fig2,
        heatmap,
        lat,
        lat_bins,
        lat_grid,
        lon,
        lon_bins,
        lon_grid,
        xedges,
        yedges,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
