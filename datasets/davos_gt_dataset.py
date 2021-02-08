import os
import numpy as np
import geopandas as gpd
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils import data_utils, viz_utils
from datasets.avalanche_dataset_base import AvalancheDatasetBase


class DavosGtDataset(AvalancheDatasetBase):
    """
    Avalanche dataset with sample corresponding to points from the ground truth data in Davos.

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
    :return pytorch dataset to be used with dataloader
    """

    def __init__(self, root_dir, gt_file, aval_file, dem_path=None, tile_size=(256, 256), bands=None,
                 means=None, stds=None):

        super().__init__(root_dir, aval_file, dem_path, tile_size, bands, means, stds)
        gt_path = os.path.join(root_dir, gt_file)
        self.gt_points = gpd.read_file(gt_path)
        self.to_tensor = ToTensor()

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

        px_offset = np.array(2 * [self.tile_size // 2])
        vrt_offset = np.array([p.x - self.ulx, self.uly - p.y])
        vrt_offset = vrt_offset / self.pixel_w - px_offset
        aval_offset = np.array([p.x - self.aval_ulx, self.aval_uly - p.y])
        aval_offset = aval_offset / self.pixel_w - px_offset

        image = data_utils.get_all_bands_as_numpy(self.vrt, vrt_offset, self.tile_size,
                                                  means=self.means, stds=self.stds, bands=self.bands)
        image = data_utils.redistribute_satellite_data(image)
        shp_image = data_utils.get_all_bands_as_numpy(self.aval_raster, aval_offset, self.tile_size)

        # add DEM after changing brightness etc but before rotating and flipping
        if self.dem:
            dem_offset = np.array([p.x - self.dem_ulx, self.dem_uly - p.y])
            dem_offset = dem_offset / self.pixel_w - px_offset
            dem_image = data_utils.get_all_bands_as_numpy(self.dem, dem_offset, self.tile_size,
                                                          means=[2100], stds=[1000])
            image = np.concatenate([image, dem_image], axis=2)

        image = self.to_tensor(image)
        shp_image = self.to_tensor(shp_image)

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

    my_dataset = DavosGtDataset(data_folder, gt_file, ava_file, tile_size=256, dem_path=None)
    dataloader = DataLoader(my_dataset, batch_size=1, shuffle=False, num_workers=2)

    for batch in iter(dataloader):
        image, shp_image = batch[0:2]
        image = image.permute(0,2,3,1)
        shp_image = shp_image.permute(0,2,3,1)

        viz_utils.plot_avalanches_by_certainty(image, shp_image, my_dataset.dem)
        input('Press key for another sample')
