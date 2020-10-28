import os
import geopandas as gpd
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal
from utils import data_utils, viz_utils
from torchvision.transforms import ToTensor
from utils.data_augmentation import RandomScaling, RandomShift
import matplotlib.pyplot as plt
import numpy as np
import torch


class DavosGtDataset(Dataset):
    """
    SLF Avalanche Dataset with sample corresponding to points from the ground truth data in davos

    :param root_dir: directory in which all data is located
    :param aval_file: shapefile name located in root_dir of the avalanches
    :param region_file: shapefile containing polygon specifying which area will be considered by the Dataset
    :param dem_path: file path of digital elevation model if it is to be used. Default: None
    :param tile_size: patch size to use for training
    :param bands: list of band indexes to read from optical images. Default None gets all
    :param certainty: Which avalanches to consider. Default: all, 1: exact, 2: estimated, 3: guessed
    :param random: whether extracted patches should be shifted randomly or centered on the avalanche
    :param means: list of means for each band in the optical imagery used for standardisation
    :param stds: list of standard deviations for each band in the optical imagery for standardisation
    :param transform: transform to apply to data. Eg. rotation, toTensor, etc.
    :return pytorch dataset to be used with dataloader
    """

    def __init__(self, root_dir, gt_file, aval_file, dem_path=None, tile_size=(256, 256), bands=None,
                 means=None, stds=None, transform=None):
        print('Creating Avalanche Dataset...')
        self.tile_size = np.array(tile_size)
        self.bands = bands
        self.means = means
        self.stds = stds
        self.transform = transform

        aval_raster_path = os.path.join(root_dir, os.path.splitext(aval_file)[0] + '.tif')
        vrt_padding = 1.5 * self.tile_size.max() # padding around vrts [m] to avoid index error when reading near edge

        # open satellite images - all tiffs found in root directory
        all_tiffs = data_utils.list_paths_in_dir(root_dir, ('.tif', '.TIF', '.img', '.IMG'))
        all_tiffs.remove(aval_raster_path)
        self.vrt = data_utils.build_padded_vrt(all_tiffs, vrt_padding)

        geo_transform = self.vrt.GetGeoTransform()
        self.pixel_w = geo_transform[1]  # pixel width eg. 1 pixel => 1.5m

        # get x and y coordinates of upper left corner
        self.ulx, self.uly, _, _ = data_utils.get_raster_extent(self.vrt)

        # get rasterised avalanches
        self.aval_raster = data_utils.build_padded_vrt(aval_raster_path, vrt_padding)
        self.aval_ulx, self.aval_uly, _, _ = data_utils.get_raster_extent(self.aval_raster)

        # get DEM if specified
        self.dem = None
        if dem_path:
            # read DEM through vrt because of errors when using multiple workers without vrt
            self.dem = data_utils.build_padded_vrt(dem_path, vrt_padding)
            self.dem_ulx, self.dem_uly, _, _ = data_utils.get_raster_extent(self.dem)

        gt_path = os.path.join(root_dir, gt_file)
        self.gt_points = gpd.read_file(gt_path)

    def __len__(self):
        return len(self.gt_points)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        :param idx: index
        :return: [image, aval, mapped, status, id] as list.
                 aval: image of ground truth
                 mapped (bool): whether it was mapped in satellite image
                 status = [1: avalanche occurred, 2: unkown, 3: no avalanche, 4: old]
                 id: id of avalanche
        """
        sample = self.gt_points.iloc[idx]
        p = sample.geometry
        correspon_aval_id = sample['ID_Lawkart']
        status = sample['Aval_Vali']

        mapped = False
        if correspon_aval_id != 0:
            mapped = True

        px_offset = self.tile_size // 2
        vrt_offset = np.array([p.x - self.ulx, self.uly - p.y])
        vrt_offset = vrt_offset / self.pixel_w - px_offset
        aval_offset = np.array([p.x - self.aval_ulx, self.aval_uly - p.y])
        aval_offset = aval_offset / self.pixel_w - px_offset

        image = data_utils.get_all_bands_as_numpy(self.vrt, vrt_offset, self.tile_size.tolist(),
                                                  means=self.means, stds=self.stds, bands=self.bands)
        shp_image = data_utils.get_all_bands_as_numpy(self.aval_raster, aval_offset, self.tile_size.tolist())

        # add DEM after changing brightness etc but before rotating and flipping
        if self.dem:
            dem_offset = np.array([p.x - self.dem_ulx, self.dem_uly - p.y])
            dem_offset = dem_offset / self.pixel_w - px_offset
            dem_image = data_utils.get_all_bands_as_numpy(self.dem, dem_offset, self.tile_size.tolist(),
                                                          means=[2800], stds=[1000])
            image = np.concatenate([image, dem_image], axis=2)

        if self.transform:
            image = self.transform(image)
            shp_image = self.transform(shp_image)

        return [image, shp_image, mapped, status, sample['Id']]


if __name__ == '__main__':
    # run test

    # home
    data_folder = '/home/patrick/ecovision/data/2018'
    ava_file = 'avalanches0118_endversion.shp'
    region_file = 'Region_Selection.shp'
    gt_file = 'Methodenvergleich2018.shp'

    # pfpc
    # data_folder = '/home/pf/pfstud/bartonp/slf_avalanches/2018'
    # ava_file = 'avalanches0118_endversion.shp'
    # region_file = 'Val_area_2018.shp'
    # dem_path="" #'/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056.tif'

    my_dataset = DavosGtDataset(data_folder, gt_file, ava_file, tile_size=[256, 256], dem_path=None)
    dataloader = DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=2)

    for batch in iter(dataloader):
        # batch = [elem.squeeze() for elem in batch]
        # viz_utils.plot_avalanches_by_certainty(*batch)
        input('Press key for another sample')
