import numpy as np
import pyproj
import rasterio
import rasterio.plot
import rioxarray as rioxr
import xarray as xr


def open_ratgeo(ratgeo_file):
    da = rioxr.open_rasterio(ratgeo_file)
    da.data[da.data == da.values.min()] = np.nan
    return da


def add_pam2rat(rat_file_path):
    # PAM (Persistent Auxiliary metadata) .aux.xml sidecar file
    with rasterio.open(rat_file_path, "r+") as dataset:
        datasetval = dataset.read()
        print("NoData Value: ", datasetval.min())
        dataset.nodata = datasetval.min()

    return
