import numpy as np
from osgeo import gdal, ogr
import geopandas as gpd
from shapely.geometry import Point
from matplotlib import pyplot as plt

def get_all_bands_as_numpy(raster, offset=(0, 0), res=None, bands=None):
    """
    Fetches bands from a raster, stacks them and returns normalised numpy array.

    :param raster: gdal raster object
    :param offset: offset tuple (x,y) in pixels from top left corner
    :param res: output size (x,y) in pixels (crops input). Default is to use the whole raster.
    :param bands: list of bands to extract. Default is all.
    :return: normalised numpy array
     """

    if res is None:
        res = (raster.RasterXSize, raster.RasterYSize)

    if bands is None:
        bands = range(1, raster.RasterCount + 1)

    band_list = []
    for band in bands:
        rb = raster.GetRasterBand(band)
        rb.SetNoDataValue(0)
        band_list.append(rb.ReadAsArray(offset[0], offset[1], res[0], res[1], buf_type=gdal.GDT_Float32))

    image = np.stack(band_list, axis=2)
    if image.max() != 0:
        image /= image.max()
    return image


def get_numpy_from_shapefile(shapefile, ref_raster, offset=(0, 0), res=None):
    """
    Rasterises shapefile and return numpy array. The shapefile should be opened with ogr.

    :param shapefile: shapefile openened with ogr
    :param ref_raster: reference raster which shapefile will be projected on.
    :param offset: offset (x,y) in pixels from the top left corner.
    :param res: output size (x, y) in pixels. Output will be cropped to this. Default is size of reference raster.
    :return: numpy array of shapefile at specified window
    """

    if res is None:
        res = (ref_raster.RasterXSize, ref_raster.RasterYSize)

    # get spatial information from reference raster
    geo_transform = ref_raster.GetGeoTransform()
    pixel_w = geo_transform[1]  # pixel width eg. 1 pixel => 1.5m
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    y_min = y_max + geo_transform[5] * ref_raster.RasterYSize
    y_res = ref_raster.RasterYSize

    # prepare raster
    shape_raster = gdal.GetDriverByName('MEM').Create('', res[0], res[1], 1, gdal.GDT_Byte)
    shape_raster.SetGeoTransform(
        (x_min + offset[0] * pixel_w, pixel_w, 0, y_min + (y_res - offset[1]) * pixel_w, 0, -pixel_w))

    # rasterise
    shape_l = shapefile.GetLayer()
    err = gdal.RasterizeLayer(shape_raster, [1], shape_l, burn_values=[1])
    if err != 0:
        print('Rasterising error: ', err)

    return shape_raster.ReadAsArray()


def get_raster_extent(raster):
    """
    Get the spatial extent of the raster in its coordinate frame
    :param raster: gdal raster object
    :return: ulx, uly, lrx, lry whereby ul is upper left and lr is lower right in meters
    """
    ulx, xres, xskew, uly, yskew, yres = raster.GetGeoTransform()
    lrx = ulx + (raster.RasterXSize * xres)
    lry = uly + (raster.RasterYSize * yres)
    return ulx, uly, lrx, lry


def generate_point_grid(region, tile_size, overlap=(0, 0)):
    """
    Generate a geopandas Geoseries object of coordinates within a region
    corresponding to the top left corner of the corrsponding patch.

    The region is expanded before checking whether points are within it such that
    it is completely covered by the points.

    :param region: shapefile opened with geopandas
    :param tile_size: size (x,y) that patches will be in meters
    :param overlap: overlap of tiles in (x,y) in meters
    :return: geopandas geoseries of coordinates as Points
    """

    spacing = (tile_size[0] - overlap[0], tile_size[1] - overlap[1])

    # generate uniform grid over the entire extent
    minx, miny, maxx, maxy = region.total_bounds
    X, Y = np.mgrid[minx:maxx + spacing[0]:spacing[0], miny:maxy + spacing[1]:spacing[1]]
    X, Y = X.ravel(), Y.ravel()
    points = gpd.GeoSeries(map(Point, zip(X, Y)))

    # slightly expand regions and filter out points outside of polygons
    expanded_region = region.buffer(max(tile_size)/4, join_style=2)
    mask = points.within(expanded_region.loc[0])
    for i in range(1, len(expanded_region)):
        mask |= points.within(expanded_region.loc[i])
    points = points.loc[mask]

    # shift coordinates such that they are in the top left of each patch
    points = points.translate(xoff=-tile_size[0] / 2, yoff=tile_size[1] / 2)

    return points


def get_avalanches_in_region(avalanches, region):
    """
    Return geoseries of all avalanches within the region
    """
    selection = avalanches.intersects(region.geometry.loc[0])
    for i in range(1, len(region)):
        selection |= avalanches.intersects(region.geometry.loc[i])

    return avalanches[selection]

