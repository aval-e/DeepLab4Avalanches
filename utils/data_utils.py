import numpy as np
from osgeo import gdal, ogr


def getAllBandsAsNumpy(raster, offset=(0, 0), res=None, bands=None):
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
        band_list.append(rb.ReadAsArray(offset[0], offset[1], res[0], res[1]))

    image = np.stack(band_list, axis=2)
    return image / image.max()


def getNumpyFromShapefile(shapefile, ref_raster, offset=(0, 0), res=None):
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
    pixel_w = geo_transform[1] # pixel width eg. 1 pixel => 1.5m
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    y_min = y_max + geo_transform[5] * ref_raster.RasterYSize
    y_res = ref_raster.RasterYSize

    # prepare raster
    shape_raster = gdal.GetDriverByName('MEM').Create('', res[0], res[1], 1, gdal.GDT_Byte)
    shape_raster.SetGeoTransform((x_min + offset[0] * pixel_w, pixel_w, 0, y_min + (y_res - offset[1]) * pixel_w, 0, -pixel_w))

    # rasterise
    shape_l = shapefile.GetLayer()
    err = gdal.RasterizeLayer(shape_raster, [1], shape_l, burn_values=[1])
    if err != 0:
        print('Rasterising error: ', err)

    return shape_raster.ReadAsArray()