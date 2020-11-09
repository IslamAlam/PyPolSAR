import numpy as np
import pyproj
import rasterio
import rasterio.plot
import rioxarray
import xarray as xr
from rioxarray import merge


def xr_reproject(da_xr, reporject=None):
    from pyproj import CRS

    dst_crs = CRS("epsg:4326")
    dd_wgs = da_xr.rio.reproject(dst_crs=dst_crs)
    # dd_wgs.attrs['_FillValue'] = np.nan
    dd_wgs.data[dd_wgs.data == np.nanmin(dd_wgs.values)] = np.nan
    # np.ma.masked_where()
    return dd_wgs


def create_mosaic(list_of_files, reporject=None):
    dem = []
    for fp in list_of_files:
        print(fp)
        dem.append(
            rioxarray.open_rasterio(fp, masked=True, resampling="average")
        )
    # Merge files
    dem_mosaic = merge.merge_arrays(dem)
    if reporject is not None:
        dem_mosaic_wgs84 = xr_reproject(dem_mosaic)
        return dem_mosaic_wgs84
    return dem_mosaic
