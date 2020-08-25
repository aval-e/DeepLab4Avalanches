import os
import glob
import geopandas as gpd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal, ogr
from utils import data_utils, viz_utils


class AvalancheDataset(Dataset):
    """
    SLF Avalanche Dataset.

    :param root_dir: directory in which all data is located
    :param image_dirs: list of image directories containing tif satellite images
    :param shape_file: shapefile name located in root_dir of the avalanches
    :param region_file: shapefile containing polygon specifying which area will be considered by the Dataset
    :param tile_size: size in meters (x,y) of a sample
    :param overlap: overlap in meters (x,y) of samples. Use this when context information is important like when only
                    start or end of avalanche is visible.
    :param transform: transform to apply to data. Eg. rotation, toTensor, etc.
    :return pytorch dataset to be used with dataloader
    """

    def __init__(self, root_dir, image_dirs, shape_file, region_file, tile_size=(1000, 1000), overlap=(0, 0),
                 transform=None):
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform

        # open satellite images
        all_tiffs = []
        for image_dir in image_dirs:
            all_tiffs += glob.glob(os.path.join(os.path.join(root_dir, image_dir), '*.TIF'))
        self.vrt = gdal.BuildVRT('/tmp/myVRT.vrt', all_tiffs)

        geo_transform = self.vrt.GetGeoTransform()
        self.pixel_w = geo_transform[1]  # pixel width eg. 1 pixel => 1.5m

        self.res = (round(self.tile_size[0] / self.pixel_w), round(self.tile_size[1] / self.pixel_w))

        # get x and y coordinates of upper left corner
        self.ulx, self.uly, _, _ = data_utils.get_raster_extent(self.vrt)

        # get avalanche shapes
        shape_path = os.path.join(root_dir, shape_file)
        self.shapes = ogr.Open(shape_path)

        # get sample points within region
        region = gpd.read_file(os.path.join(root_dir, region_file))
        self.points = data_utils.generate_point_grid(region, tile_size, overlap)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        :param idx: index
        :return: [image, rasterised avalanches] as list
        """
        coord = self.points.iloc[idx]

        offset = ((coord.x - self.ulx) / self.pixel_w, (self.uly - coord.y) / self.pixel_w)

        image = data_utils.get_all_bands_as_numpy(self.vrt, offset, self.res)
        shp_image = data_utils.get_numpy_from_shapefile(self.shapes, self.vrt, offset, self.res)

        return [image, shp_image]


if __name__ == '__main__':
    # run test

    # data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
    data_folder = '/home/patrick/ecovision/data'
    tif_folder = 'Spot6_Ortho_2_3_3_4_5'
    ava_file = 'avalanches0119_endversion.shp'
    # region_file = 'Region_Selection.shp'
    region_file = 'Multiple_regions.shp'

    my_dataset = AvalancheDataset(data_folder, [tif_folder], ava_file, region_file)
    dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)
    batch = next(dataiter)

    batch = [elem.squeeze() for elem in batch]
    viz_utils.viz_sample(batch)