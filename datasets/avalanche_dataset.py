import os
import geopandas as gpd
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
from utils import data_utils, viz_utils
from math import log, ceil
from random import randint
import numpy as np
from torchvision.transforms import ToTensor


class AvalancheDataset(Dataset):
    """
    SLF Avalanche Dataset. Samples are taken from avalanches within specified region

    :param root_dir: directory in which all data is located
    :param aval_file: shapefile name located in root_dir of the avalanches
    :param region_file: shapefile containing polygon specifying which area will be considered by the Dataset
    :param dem_path: file path of digital elevation model if it is to be used. Default: None
    :param tile_size: patch size to use for training
    :param certainty: Which avalanches to consider. Default: all, 1: exact, 2: estimated, 3: guessed
    :param random: whether extracted patches should be shifted randomly or centered on the avalanche
    :param transform: transform to apply to data. Eg. rotation, toTensor, etc.
    :return pytorch dataset to be used with dataloader
    """

    def __init__(self, root_dir, aval_file, region_file, dem_path=None, tile_size=(512, 512), certainty=None,
                 random=True, transform=None):
        self.tile_size = np.array(tile_size)
        self.random = random
        self.transform = transform

        aval_raster_path = os.path.join(root_dir, os.path.splitext(aval_file)[0] + '.tif')

        # open satellite images - all tiffs found in root directory
        all_tiffs = data_utils.list_paths_in_dir(root_dir, ('.tif', '.TIF', '.img', '.IMG'))
        all_tiffs.remove(aval_raster_path)
        self.vrt = gdal.BuildVRT('', all_tiffs)

        geo_transform = self.vrt.GetGeoTransform()
        self.pixel_w = geo_transform[1]  # pixel width eg. 1 pixel => 1.5m

        # get x and y coordinates of upper left corner
        self.ulx, self.uly, _, _ = data_utils.get_raster_extent(self.vrt)

        # get avalanche shapes with geopandas
        region = gpd.read_file(os.path.join(root_dir, region_file))
        aval_path = os.path.join(root_dir, aval_file)
        self.avalanches = gpd.read_file(aval_path)
        self.avalanches = data_utils.get_avalanches_in_region(self.avalanches, region)

        # get rasterised avalanches
        self.aval_raster = gdal.BuildVRT('', aval_raster_path)
        self.aval_ulx, self.aval_uly, _, _ = data_utils.get_raster_extent(self.aval_raster)

        if certainty:
            self.avalanches = self.avalanches[self.avalanches.aval_shape <= certainty]

        # get DEM if specified
        self.dem = None
        if dem_path:
            # read DEM through vrt because of errors when using multiple workers without vrt
            self.dem = gdal.BuildVRT('', dem_path)
            self.dem_ulx, self.dem_uly, _, _ = data_utils.get_raster_extent(self.dem)

    def __len__(self):
        return len(self.avalanches)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        :param idx: index
        :return: [image, rasterised avalanches] as list
        """
        aval = self.avalanches.iloc[idx]
        bbox = aval.geometry.bounds

        # calculate pixel coords, width and height of patch
        res_aval_px = np.array((bbox[2] - bbox[0], bbox[3] - bbox[1]))
        res_aval_px = np.ceil(res_aval_px / self.pixel_w)
        size_diff = self.tile_size - res_aval_px

        # move avalanche to center or augment position around center if random enabled
        px_offset = np.array([0, 0])
        if self.random:
            if size_diff[0] > 0:
                px_offset[0] = randint(0, size_diff[0])
            else:
                px_offset[0] = randint(size_diff[0], 0)
            if size_diff[1] > 0:
                px_offset[1] = randint(0, size_diff[1])
            else:
                px_offset[1] = randint(size_diff[1], 0)
        else:
            px_offset = size_diff // 2
        vrt_offset = np.array([bbox[0] - self.ulx, self.uly - bbox[3]])
        vrt_offset = vrt_offset / self.pixel_w - px_offset
        aval_offset = np.array([bbox[0] - self.aval_ulx, self.aval_uly - bbox[3]])
        aval_offset = aval_offset / self.pixel_w - px_offset

        image = data_utils.get_all_bands_as_numpy(self.vrt, vrt_offset, self.tile_size.tolist(), normalise=True)
        shp_image = data_utils.get_all_bands_as_numpy(self.aval_raster, aval_offset, self.tile_size.tolist(), normalise=False)

        if self.dem:
            dem_offset = np.array([bbox[0] - self.dem_ulx, self.dem_uly - bbox[3]])
            dem_offset = dem_offset / self.pixel_w - px_offset
            dem_image = data_utils.get_all_bands_as_numpy(self.dem, dem_offset, self.tile_size.tolist(), normalise=False)
            dem_image = dem_image / 5000  # normalisation for alps - max <5000 meters
            image = np.concatenate([image, dem_image], axis=2)

        if self.transform:
            image = self.transform(image)
            shp_image = self.transform(shp_image)

        return [image, shp_image]


if __name__ == '__main__':
    # run test

    # home
    # data_folder = '/home/patrick/ecovision/data/2019'
    # ava_file = 'avalanches0119_endversion.shp'
    # region_file = 'Multiple_regions.shp'

    # pfpc
    data_folder = '/home/pf/pfstud/bartonp/slf_avalanches/2018'
    ava_file = 'avalanches0118_endversion.shp'
    region_file = 'Val_area_2018.shp'
    dem_path="" #'/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056.tif'

    my_dataset = AvalancheDataset(data_folder, ava_file, region_file, dem_path=dem_path)
    dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True, num_workers=2)

    dataiter = iter(dataloader)
    batch = next(dataiter)

    batch = [elem.squeeze() for elem in batch]
    viz_utils.overlay_and_plot_avalanches_by_certainty(*batch)
