import os
import geopandas as gpd
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
from utils import data_utils, viz_utils, utils
from torchvision.transforms import ToTensor
from utils.data_augmentation import RandomScaling, RandomShift
import matplotlib.pyplot as plt
import numpy as np
import torch


class AvalancheDatasetPoints(Dataset):
    """
    SLF Avalanche Dataset. Samples chosen intelligently to avoid overlaps and cover larger avalanches

    :param root_dir: directory in which all data is located
    :param aval_file: shapefile name located in root_dir of the avalanches
    :param region_file: shapefile containing polygon specifying which area will be considered by the Dataset
    :param dem_path: file path of digital elevation model if it is to be used. Default: None
    :param tile_size: patch size to use for training
    :param bands: list of band indexes to read from optical images. Default None gets all
    :param certainty: Which avalanches to consider. Default: all, 1: exact, 2: estimated, 3: guessed
    :param batch_augm (int): whether to perform batch augmentation and how many samples to return
    :param random: whether extracted patches should be shifted randomly or centered on the avalanche
    :param means: list of means for each band in the optical imagery used for standardisation
    :param stds: list of standard deviations for each band in the optical imagery for standardisation
    :param transform: transform to apply to data. Eg. rotation, toTensor, etc.
    :return pytorch dataset to be used with dataloader
    """

    def __init__(self, root_dir, aval_file, region_file, dem_path=None, tile_size=512, bands=None,
                 certainty=None, batch_augm=0,
                 random=True, means=None, stds=None, transform=None):
        print('Creating Avalanche Dataset...')
        self.tile_size = tile_size
        self.bands = bands
        self.random = random
        self.means = means
        self.stds = stds
        self.ba = batch_augm if batch_augm > 1 else 1
        self.transform = transform

        self.rand_shift = RandomShift(0.2)
        self.rand_scale = RandomScaling(0.3)
        self.rand_shift_dem = RandomShift(1.0)

        aval_raster_path = os.path.join(root_dir, os.path.splitext(aval_file)[0] + '.tif')
        vrt_padding = 1.5 * self.tile_size  # padding around vrts [m] to avoid index error when reading near edge

        # open satellite images - all tiffs found in root directory
        all_tiffs = data_utils.list_paths_in_dir(root_dir, ('.tif', '.TIF', '.img', '.IMG'))
        all_tiffs.remove(aval_raster_path)
        self.vrt = data_utils.build_padded_vrt(all_tiffs, vrt_padding)

        geo_transform = self.vrt.GetGeoTransform()
        self.pixel_w = geo_transform[1]  # pixel width eg. 1 pixel => 1.5m

        # get x and y coordinates of upper left corner
        self.ulx, self.uly, _, _ = data_utils.get_raster_extent(self.vrt)

        # get avalanche shapes with geopandas
        region = gpd.read_file(os.path.join(root_dir, region_file))
        aval_path = os.path.join(root_dir, aval_file)
        self.avalanches = gpd.read_file(aval_path)
        self.avalanches = data_utils.get_avalanches_in_region(self.avalanches, region)
        if certainty:
            self.avalanches = self.avalanches[self.avalanches.aval_shape <= certainty]
        self.sample_points = data_utils.generate_sample_points(self.avalanches, region, self.tile_size)

        # get rasterised avalanches
        self.aval_raster = data_utils.build_padded_vrt(aval_raster_path, vrt_padding)
        self.aval_ulx, self.aval_uly, _, _ = data_utils.get_raster_extent(self.aval_raster)

        # get DEM if specified
        self.dem = None
        if dem_path:
            # read DEM through vrt because of errors when using multiple workers without vrt
            self.dem = data_utils.build_padded_vrt(dem_path, vrt_padding)
            self.dem_ulx, self.dem_uly, _, _ = data_utils.get_raster_extent(self.dem)

    def __len__(self):
        return len(self.sample_points)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        :param idx: index
        :return: [image, rasterised avalanches] as list
        """
        p = self.sample_points.iloc[idx]

        samples = []
        for sample in range(self.ba):
            px_offset = np.array(2 * [self.tile_size // 2])
            if self.random:
                max_diff = self.tile_size // 3
                px_offset += np.random.randint(-max_diff, max_diff, 2)
            vrt_offset = np.array([p.x - self.ulx, self.uly - p.y])
            vrt_offset = vrt_offset / self.pixel_w - px_offset
            aval_offset = np.array([p.x - self.aval_ulx, self.aval_uly - p.y])
            aval_offset = aval_offset / self.pixel_w - px_offset

            image = data_utils.get_all_bands_as_numpy(self.vrt, vrt_offset, self.tile_size,
                                                      means=self.means, stds=self.stds, bands=self.bands)
            shp_image = data_utils.get_all_bands_as_numpy(self.aval_raster, aval_offset, self.tile_size)

            # augment one of brightness and contrast
            if self.random:
                image = self.rand_shift(image)
                image = self.rand_scale(image)

            # add DEM after changing brightness etc but before rotating and flipping
            if self.dem:
                dem_offset = np.array([p.x - self.dem_ulx, self.dem_uly - p.y])
                dem_offset = dem_offset / self.pixel_w - px_offset
                dem_image = data_utils.get_all_bands_as_numpy(self.dem, dem_offset, self.tile_size,
                                                              means=[2800], stds=[1000])
                if self.random:
                    dem_image = self.rand_shift_dem(dem_image)
                image = np.concatenate([image, dem_image], axis=2)

            if self.transform:
                array = np.concatenate([image, shp_image], axis=2)
                array = self.transform(array)

                if torch.is_tensor(array):
                    image = array[:-1, :, :]
                    shp_image = array[-1:, :, :]
                else:
                    image = array[:, :, :-1]
                    shp_image = array[:, :, -1]
            samples.append([image, shp_image])

        return samples if self.ba > 1 else samples[0]


if __name__ == '__main__':
    # run test

    # home
    # data_folder = '/home/patrick/ecovision/data/2018'
    # ava_file = 'avalanches0118_endversion.shp'
    # region_file = 'Region_Selection.shp'
    # dem_path = None #'/home/patrick/ecovision/data/2018/avalanches0118_endversion.tif'

    # pfpc
    data_folder = '/home/pf/pfstud/bartonp/slf_avalanches/2018'
    ava_file = 'avalanches0118_endversion.shp'
    region_file = 'Val_area_2018.shp'
    dem_path='/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056.tif'

    my_dataset = AvalancheDatasetPoints(data_folder, ava_file, region_file, tile_size=256, dem_path=dem_path,
                                        random=True, batch_augm=1)
    dataloader = DataLoader(my_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.ba_collate_fn)

    for batch in iter(dataloader):
        viz_utils.plot_avalanches_by_certainty(*batch, dem=my_dataset.dem)
        input('Press key for another sample')
