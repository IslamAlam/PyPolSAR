import matplotlib.pyplot as plt



def scale_bar(ax, length=None, location=(0.5, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    from cartopy import crs as ccrs
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly, approx=True)
    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length: 
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)        
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length) 

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]
    #Plot the scalebar
    ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc,
            horizontalalignment='center', verticalalignment='bottom')
    
    
def MapPlot(ax=None, extend=None, scale_bar_plot=True, scale_bar_length=10,  scale_bar_location=(0.5, 0.05), scale_bar_linewidth=3):
    from cartopy import crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    
    # Set Projection to EPSG:3857 (aka Pseudo-Mercator, Spherical Mercator or Web Mercator) is the coordinate system used by Google Maps and web mapping application.
    crs_epsg = ccrs.epsg('3857')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_epsg},
                        figsize=(10, 10))
        
    # Make the CartoPy plot
    ax = plt.axes(projection=crs_epsg)
    gl = ax.gridlines(draw_labels=True, alpha=0.2)
    gl.top_labels = gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if extend is None:
        ax.set_extent([0, 180, 0,  90], crs=ccrs.PlateCarree(central_longitude=0))
    else:
        ax.set_extent(extend, crs=ccrs.PlateCarree(central_longitude=0))
        
    if scale_bar_plot == True:
        scale_bar(ax, length=scale_bar_length, location=scale_bar_location, linewidth=scale_bar_linewidth)

        
    return ax

def MapDataPlot(ax=None, extend=None, scale_bar_plot=True, scale_bar_length=10,  scale_bar_location=(0.5, 0.05), scale_bar_linewidth=3):
    from cartopy import crs as ccrs
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    
    # Set Projection to EPSG:3857 (aka Pseudo-Mercator, Spherical Mercator or Web Mercator) is the coordinate system used by Google Maps and web mapping application.
    crs_epsg = ccrs.epsg('3857')
    if ax is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_epsg},
                        figsize=(10, 10))
        
    # Make the CartoPy plot
    ax = plt.axes(projection=crs_epsg)

    if extend is None:
        ax.set_extent([0, 180, 0,  90], crs=ccrs.PlateCarree(central_longitude=0))
    else:
        ax.set_extent(extend, crs=ccrs.PlateCarree(central_longitude=0))
        
    if scale_bar_plot == True:
        scale_bar(ax, length=scale_bar_length, location=scale_bar_location, linewidth=scale_bar_linewidth)

        
    return ax