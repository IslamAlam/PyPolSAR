import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import rasterio.plot
import xarray as xr


def plot_dem(longitude, latitude, topo, vmin=-50, vcenter=0, vmax=250):

    fig, ax = plt.subplots(constrained_layout=True, figsize=(12, 12))
    # make a colormap that has land and ocean clearly delineated and of the
    # same length (256 + 256)
    colors_undersea = plt.cm.terrain(np.linspace(0, 0.17, 256))
    colors_land = plt.cm.terrain(np.linspace(0.25, 1, 256))
    all_colors = np.vstack((colors_undersea, colors_land))
    terrain_map = colors.LinearSegmentedColormap.from_list(
        "terrain_map", all_colors
    )

    # make the norm:  Note the center is offset so that the land has more
    # dynamic range:
    divnorm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    pcm = ax.pcolormesh(
        longitude,
        latitude,
        topo,
        rasterized=True,
        norm=divnorm,
        cmap=terrain_map,
        alpha=1,
        shading="auto",
        edgecolors="none",
        facecolors="b",
    )  # shading='auto', edgecolors='face') ,shading='gouraud'
    ax.set_xlabel("Lon $[^o E]$")
    ax.set_ylabel("Lat $[^o N]$")
    ax.set_aspect(1 / np.cos(np.deg2rad(latitude.mean())))
    fig.colorbar(pcm, shrink=0.6, extend="both", label="Elevation [m]")
    return fig
