import numpy as np
from scipy import ndimage
import pyproj
import rasterio as rio
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


def write_raster(filename, arr, transform):

    profile = rio.profiles.Profile( )
    profile.update(dtype=arr.dtype, count=1, compress='lzw', driver='GTiff', crs='+proj=latlong', nodata=np.min(arr),
                           width=arr.shape[1], height=arr.shape[0], transform=transform)
    with rio.open(filename, 'w', **profile) as dst:
        dst.write(arr, 1)
        

def write_sr2latlon_raster(filename, 
                           rgi_sr_arr, 
                           lut_sr2latlon_az_arr, 
                           lut_sr2latlon_rg_arr,
                           lon_min, 
                           lon_max,
                           lat_min,
                           lat_max, rescale=None, order=1, cval=np.nan, mode='constant', *args, **kwargs):
    
    assert lut_sr2latlon_az_arr.shape == lut_sr2latlon_rg_arr.shape
    if rescale is not None:
        from skimage.transform import resize
        rescale_factor = rescale

        lut_sr2latlon_az_arr = resize(lut_sr2latlon_az_arr, (int(lut_sr2latlon_az_arr.shape[0] // rescale_factor), int(lut_sr2latlon_az_arr.shape[1] // rescale_factor)),
                               anti_aliasing=True)

        lut_sr2latlon_rg_arr = resize(lut_sr2latlon_rg_arr, (int(lut_sr2latlon_rg_arr.shape[0] // rescale_factor), int(lut_sr2latlon_rg_arr.shape[1] // rescale_factor)),
                               anti_aliasing=True)

    # Notice south is lat_max and north is lat_min due to the affine transfrom is different from gdal and python
    transform = rio.transform.from_bounds(west=lon_min, south=lat_max, east=lon_max, north=lat_min, width=lut_sr2latlon_rg_arr.shape[1], height=lut_sr2latlon_rg_arr.shape[0])
    wrap = ndimage.map_coordinates(rgi_sr_arr, [lut_sr2latlon_az_arr, lut_sr2latlon_rg_arr], order=order, cval=cval, mode=mode, *args, **kwargs)
    
    write_raster(filename=filename, arr=wrap, transform=transform,)
    
def write_sr2geo_raster(save_filename, rgi_sr_arr, lut_sr2geo_az_path, lut_sr2geo_rg_path, rescale=None, order=1, cval=np.nan, mode='constant', *args, **kwargs):
    
    lut_sr2geo_az = rioxarray.open_rasterio(lut_sr2geo_az_path)
    lut_sr2geo_rg = rioxarray.open_rasterio(lut_sr2geo_rg_path)
     
    wrap = ndimage.map_coordinates(rgi_sr_arr, [lut_sr2geo_az, lut_sr2geo_rg], order=order, cval=cval, mode=mode, *args, **kwargs)
    temp = lut_sr2geo_az
    temp.data = wrap
    temp.rio.set_nodata(np.nan)
    keys_to_remove = ('STATISTICS_MAXIMUM', 'STATISTICS_MEAN', 'STATISTICS_MINIMUM', 'STATISTICS_STDDEV', 'STATISTICS_VALID_PERCENT', 'description',)
    values_removed = [temp.attrs.pop(key, None) for key in keys_to_remove]
    temp.rio.to_raster(save_filename, )

    
    