import os
import torch
import numpy as np
from osgeo import gdal, ogr
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
from matplotlib import pyplot as plt
import affine
import rasterio
import rasterio.features
import random


def list_paths_in_dir(root_dir, file_endings=None):
    """
    Get a list of paths to files within a directory and subdirectories

    :param root_dir: directory to search
    :param file_endings: list of file endings to consider. Default is all
    """
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        if file_endings:
            filenames = [f for f in filenames if f.endswith(file_endings)]
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files


def build_padded_vrt(sources, padding=1000):
    """ Creates an inmemory gdal VRT from a list of sources with additional padding
    :param sources: list of paths to raster files
    :param padding: amount of padding in meters around the vrt to avoid read errors
    :returns: a gdal vrt object
    """
    vrt = gdal.BuildVRT('', sources)
    extent = get_raster_extent(vrt)
    extent = extent[0] - padding, extent[3] - padding, extent[2] + padding, extent[1] + padding
    vrt = gdal.BuildVRT('', sources, options=gdal.BuildVRTOptions(outputBounds=extent, resolution='highest'))
    return vrt


def get_all_bands_as_numpy(raster, offset=(0, 0), res=None, bands=None, means=None, stds=None):
    """
    Fetches bands from a raster, stacks them and returns normalised numpy array.
    A list of means and standard deviations the same length as the number of bands can be passed to standardise each
    one individually. To standardise all bands with the same mean and std pass a list of length 1

    :param raster: gdal raster object
    :param offset: offset tuple (x,y) in pixels from top left corner
    :param res: output size in pixels (crops input). Default is to use the whole raster.
    :param bands: list of bands to extract. Default is all.
    :param means: list of means for each band to standardise
    :param stds: list of standard deviations for each band to standardise
    :return: standardised numpy array
     """

    if res is None:
        res = (raster.RasterXSize, raster.RasterYSize)
    else:
        res = (res, res)

    if bands is None:
        bands = range(1, raster.RasterCount + 1)

    # standardise all bands the same if only one value is given
    if means and len(means) == 1:
        means = raster.RasterCount * means
    if stds and len(stds) == 1:
        stds = raster.RasterCount * stds

    band_list = []
    i = 0
    for band in bands:
        rb = raster.GetRasterBand(band)
        rb.SetNoDataValue(means[i] if means else 0)
        arr = rb.ReadAsArray(offset[0], offset[1], res[0], res[1], buf_type=gdal.GDT_Float32)

        # Standardise
        if means:
            arr -= means[i]
        if stds:
            arr /= stds[i]

        band_list.append(arr)
        arr = None
        rb = None
        i += 1

    image = np.stack(band_list, axis=2)

    return image


def redistribute_satellite_data(image):
    """ Redistributes the input data to spread it more evenly.

    The input data was found to bi bimodal with a large spike on the negative side of the standardised data
    from the shadows. Multiplying negative values by 3 as well as by themselves was found to result in a better spread
    of data.
    """
    image[image < 0] = -3 * image[image < 0] ** 2
    return image


def undo_redistribute_satellite_data(image):
    """ Reverses effects of redistribute_satellite_data for visualisation"""
    image[image < 0] = -torch.sqrt(-image[image < 0]/3)
    return image


def get_numpy_from_ogr_shapefile(shapefile, ref_raster, offset=(0, 0), res=None):
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


def rasterise_geopandas(dataset, tile_size, offset, burn_val=1, res=1.5, individual=False):
    """
    Rasterise geopandas dataset and return numpy array

    :param dataset: geopandas dataset to be rasterised
    :param tile_size: size of the output patch
    :param offset: position of top left corner. Make sure this is in the correct CRS
    :param burn_val: value to write where there is a shape. Background is zero. If string, it is interpreted as an attribute
    :param res: resolution of raster (meters per pixel)
    :param individual: whether to rasterise each avalanche into a separate channel
    """
    transform = affine.Affine(res, 0, offset[0], 0, -res, offset[1])

    # first get subset of avalanches intersecting patch before rasterising them
    subset = dataset.cx[offset[0]: offset[0] + res * tile_size[0], offset[1] + -res * tile_size[1]: offset[1]:]
    if len(subset) == 0:
        return np.zeros(tile_size)

    if isinstance(burn_val, str):
        shapes = ((sample.geometry, sample[burn_val]) for _, sample in subset.iterrows())
    else:
        shapes = ((geom, value) for geom, value in zip(subset.geometry, len(subset) * [burn_val]))
    
    raster = []
    if individual:
        for shape in shapes:
            raster.append(rasterio.features.rasterize([shape], tile_size, transform=transform, dtype=np.single))
        raster = np.stack(raster, axis=2)
    else:
        raster = rasterio.features.rasterize(shapes, tile_size, transform=transform, dtype=np.single)
    return raster


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


def labels_to_mask(labels):
    """ Convert image labels to mask of zeros and ones. Returns binary mask of avalanches ignoring their certainty"""
    return (labels != 0).float()


def get_avalanches_in_region(avalanches, region):
    """
    Return geoseries of all avalanches within the region
    """
    selection = avalanches.intersects(region.geometry.loc[0])
    for i in range(1, len(region)):
        selection |= avalanches.intersects(region.geometry.loc[i])

    return avalanches[selection]


def generate_point_grid(region, tile_size, overlap=0):
    """
    Generate a geopandas Geoseries object of coordinates within a region
    corresponding to the top left corner of the corrsponding patch.

    The region is expanded before checking whether points are within it such that
    it is completely covered by the points.

    :param region: shapefile opened with geopandas
    :param tile_size: size that patches will be in meters
    :param overlap: overlap of tiles in meters
    :return: geopandas geoseries of coordinates as Points
    """

    spacing = tile_size - overlap

    # generate uniform grid over the entire extent
    minx, miny, maxx, maxy = region.total_bounds
    X, Y = np.mgrid[minx:maxx + spacing:spacing, miny:maxy + spacing:spacing]
    X, Y = X.ravel(), Y.ravel()
    points = gpd.GeoSeries(map(Point, zip(X, Y)))

    # slightly expand regions and filter out points outside of polygons
    expanded_region = region.buffer(tile_size/4, join_style=2)
    mask = points.within(expanded_region.loc[0])
    for i in range(1, len(expanded_region)):
        mask |= points.within(expanded_region.loc[i])
    points = points.loc[mask]

    # shift coordinates such that they are in the top left of each patch
    points = points.translate(xoff=-tile_size*3/4, yoff=tile_size*3/4)

    return points


def generate_sample_points(avalanches, region, tile_size, no_aval_ratio=0.05, n=200):
    """ Intelligently choose samples such that there is no overlap but large avalanches are covered
        Also add samples with no avalanche present

        :param avalanches: geopandas geoseries of avalanche polygons
        :param region: region in which samples are contained as geoseries
        :param tile_size: size of one sample
        :param no_aval_ratio: ratio of samples to add with no avalanche [0-1]
        :param n: number of neighbour avalanches (in geoseries order) to consider when checking distance
        :returns: geoseries of sample Points
    """
    sample_points = gpd.GeoSeries()

    # add point for each avalanche, multiple points for large avalanches
    for i in tqdm(range(0, len(avalanches)), desc="Generating Points in avalanches"):
        aval = avalanches.iloc[i]
        diff = aval.geometry
        while not diff.is_empty:
            # get point inside avalanche - not random but not centroid either
            p = diff.representative_point()

            # get difference of avalanche and square around sample point
            diff = diff.difference(p.buffer(tile_size, cap_style=3))

            # only add point if it is not too close to another (could be too close to another avalanche)
            if not (sample_points.iloc[-n:].distance(p) < tile_size).any():
                sample_points = sample_points.append(gpd.GeoSeries(p))

    # add points with no avalanche
    for i in tqdm(range(0, int(no_aval_ratio * len(avalanches))), desc='Generating points with no avalanches'):
        for j in range(0, 100): # max 100 tries
            p = get_random_point_in_region(region)

            # only add point if far enough from all avalanches
            if not (sample_points.distance(p) < tile_size).any():
                sample_points = sample_points.append(gpd.GeoSeries(p))
                break

    return sample_points


def get_random_point_in_region(region):
    """ Sample a random point within a region
        :param region: geoseries with one or multiple polygons
        :returns: random point within region
    """
    minx, miny, maxx, maxy = region.total_bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if region.contains(p).any():
            return p

