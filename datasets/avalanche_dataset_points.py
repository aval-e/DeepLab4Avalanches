import os
from argparse import ArgumentParser
import geopandas as gpd
from torch.utils.data import DataLoader
from utils import data_utils, viz_utils, utils
from datasets.avalanche_dataset_base import AvalancheDatasetBase
from torchvision.transforms import ToTensor
from utils.data_augmentation import RandomScaling, RandomShift, RandomHorizontalFlip, RandomRotation
import matplotlib.pyplot as plt
import numpy as np


class AvalancheDatasetPoints(AvalancheDatasetBase):
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
    :param means: list of means for each band in the optical imagery used for standardisation
    :param stds: list of standard deviations for each band in the optical imagery for standardisation
    :param random: whether extracted patches should be shifted randomly or centered on the avalanche
    :param hflip_p: probability of a horizontal flip
    :param rand_rot: max angle in degrees by which to rotate randomly
    :return pytorch dataset to be used with dataloader
    """

    def __init__(self, root_dir, aval_file, region_file, dem_path=None, tile_size=512, bands=None,
                 certainty=None, batch_augm=0, means=None, stds=None,
                 random=True, hflip_p=0, rand_rot=0):

        super().__init__(root_dir, aval_file, dem_path, tile_size, bands, means, stds)
        self.random = random
        self.ba = batch_augm if batch_augm > 1 else 1

        self.to_tensor = ToTensor()
        self.rand_rotation = RandomRotation(rand_rot)
        self.rand_flip = RandomHorizontalFlip(hflip_p)
        self.rand_shift = RandomShift(0.2)
        self.rand_scale = RandomScaling(0.3)
        self.rand_shift_dem = RandomShift(1.0)

        # get avalanche shapes with geopandas
        region = gpd.read_file(os.path.join(root_dir, region_file))
        aval_path = os.path.join(root_dir, aval_file)
        self.avalanches = gpd.read_file(aval_path)
        self.avalanches = data_utils.get_avalanches_in_region(self.avalanches, region)
        if certainty:
            self.avalanches = self.avalanches[self.avalanches.aval_shape <= certainty]
        self.sample_points = data_utils.generate_sample_points(self.avalanches, region, self.tile_size)

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
            mask = data_utils.get_all_bands_as_numpy(self.aval_raster, aval_offset, self.tile_size)

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

            # Apply transforms
            angle = self.rand_rotation.get_param()
            image = self.rand_rotation(image, angle)
            mask = self.rand_rotation(mask, angle)

            image = self.to_tensor(image)
            mask = self.to_tensor(mask)
            
            ph = self.rand_flip.get_param()
            image = self.rand_flip(image, ph)
            mask = self.rand_flip(mask, ph)

            samples.append((image, mask))

        return samples if self.ba > 1 else samples[0]

    @staticmethod
    def add_argparse_args(parent_parser):
        # add all dataset hparams to argparse
        parser = super(AvalancheDatasetPoints, AvalancheDatasetPoints).add_argparse_args(parent_parser)
        parser = ArgumentParser(parents=[parser], add_help=False)
        parser.add_argument('--batch_augm', type=int, default=0, help='the amount of batch augmentation to use')
        parser.add_argument('--aval_certainty', type=int, default=None,
                            help='Which avalanche certainty to consider. 1: exact, 2: estimated, 3: guessed')

        parser.add_argument('--hflip_p', type=float, default=0, help='probability of horizontal flip')
        parser.add_argument('--rand_rotation', type=float, default=0, help='max random rotation in degrees')
        return parser


if __name__ == '__main__':
    # run test

    # home
    # data_folder = '/home/patrick/ecovision/data/2018'
    # ava_file = 'avalanches0118_endversion.shp'
    # region_file = 'Region_Selection.shp'
    # dem_path = '/home/patrick/ecovision/data/2018/avalanches0118_endversion.tif'

    # hard drive
    username = 'bartonp'
    year = '19'
    data_folder = '/media/' + username + '/Seagate Expansion Drive/SLF_Avaldata/20' + year
    ava_file = 'avalanches01' + year + '_endversion.shp'
    region_file = 'Val_area_20' + year + '.shp'
    dem_path = None

    # pfpc
    # data_folder = '/home/pf/pfstud/bartonp/slf_avalanches/2018'
    # ava_file = 'avalanches0118_endversion.shp'
    # region_file = 'Val_area_2018.shp'
    # dem_path='/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056.tif'

    my_dataset = AvalancheDatasetPoints(data_folder, ava_file, region_file, tile_size=256, dem_path=dem_path,
                                        random=True, batch_augm=1, rand_rot=180)
    dataloader = DataLoader(my_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.ba_collate_fn)

    for batch in iter(dataloader):
        image, shp_image = batch
        image = image.permute(0, 2, 3, 1)
        shp_image = shp_image.permute(0, 2, 3, 1)

        viz_utils.plot_avalanches_by_certainty(image, shp_image, dem=my_dataset.dem)
        input('Press key for another sample')
