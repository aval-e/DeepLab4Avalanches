import numpy as np
from osgeo import gdal, ogr


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
        band_list.append(rb.ReadAsArray(offset[0], offset[1], res[0], res[1]))

    image = np.stack(band_list, axis=2)
    return image / image.max()


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
    :return: ulx, uly, lrx, lry whereby ul is upper left and lr is lower right
    """
    ulx, xres, xskew, uly, yskew, yres = raster.GetGeoTransform()
    lrx = ulx + (raster.RasterXSize * xres)
    lry = uly + (raster.RasterYSize * yres)
    return ulx, uly, lrx, lry


def generate_point_grid(extent, step=(1, 1)):
    """
    Generate a a multipoint object of coordinates on a grid from which to sample patches for dataset

    :param extent: (ulx, uly, lrx, lry) as retrieved from get_raster_extent
    :param step: distance between multipoint (x,y)
    :return: ogr multipoint geometry object of coordinates
    """

    ulx, uly, lrx, lry = extent
    X, Y = np.mgrid[ulx:lrx:step[0], uly:lry:step[1]]
    coords = np.vstack([X.ravel(), Y.ravel()])

    multipoint = ogr.Geometry(ogr.wkbMultiPoint)
    for i in range(coords.shape[1]):
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(coords[0, i], coords[1, i])
        multipoint.AddGeometry(point)

    memDriver = ogr.GetDriverByName('Memory')

    # Create the output shapefile
    data_source = memDriver.CreateDataSource('memData')
    out_layer = data_source.CreateLayer("coords", geom_type=ogr.wkbMultiPoint)

    # Add an ID field
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    out_layer.CreateField(idField)

    # Create the feature and set values
    featureDefn = out_layer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    feature.SetGeometry(multipoint)
    feature.SetField("id", 1)
    out_layer.CreateFeature(feature)
    feature = None

    # Save and close DataSource
    # data_source = None

    return out_layer
