import os
import numpy as np
import geopandas as gpd
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal, ogr
from utils import data_utils, viz_utils
from torchvision.transforms import ToTensor


class AvalancheDatasetGrid(Dataset):
    """
    Avalanche Dataset with samples taken on a grid within a specified region

    :param root_dir: directory in which all data is located
    :param region_file: shapefile containing polygon specifying which area will be considered by the Dataset
    :param dem_path: DEM file path if DEM is to be used.
    :param aval_path: shapefile path of the ground truth avalanches if available
    :param tile_size: size in meters of a sample
    :param overlap: overlap in meters of samples. Use this when context information is important like when only
                    start or end of avalanche is visible.
    :param bands: list of band indexes to read from optical images. Default None gets all
    :param means: list of means for each band in the optical imagery used for standardisation
    :param stds: list of standard deviations for each band in the optical imagery for standardisation
    :return pytorch dataset to be used with dataloader
    """

    def __init__(self, root_dir, region_file, dem_path='', aval_path='', tile_size=1000, overlap=0, bands=None,
                 means=None, stds=None):
        self.tile_size = tile_size
        self.overlap = overlap
        self.aval_path = aval_path
        self.bands = bands
        self.means = means
        self.stds = stds

        gdal.SetCacheMax(134217728) # 134Mb to limit memory usage on leonhard

        # open satellite images - all tiffs found in root directory
        all_tiffs = data_utils.list_paths_in_dir(root_dir, ('.tif', '.TIF', '.img', '.IMG'))
        self.vrt = data_utils.build_padded_vrt(all_tiffs, tile_size)

        geo_transform = self.vrt.GetGeoTransform()
        self.pixel_w = geo_transform[1]  # pixel width eg. 1 pixel => 1.5m
        # get x and y coordinates of upper left corner
        self.ulx, self.uly, _, _ = data_utils.get_raster_extent(self.vrt)

        if aval_path:
            # get rasterised avalanches
            self.aval_raster = data_utils.build_padded_vrt(aval_path, tile_size)
            self.aval_ulx, self.aval_uly, _, _ = data_utils.get_raster_extent(self.aval_raster)

        # get DEM if specified
        self.dem = None
        if dem_path:
            # read DEM through vrt because of errors when using multiple workers without vrt
            self.dem = data_utils.build_padded_vrt(dem_path, tile_size)
            self.dem_ulx, self.dem_uly, _, _ = data_utils.get_raster_extent(self.dem)

        # get sample points within region
        region = gpd.read_file(os.path.join(root_dir, region_file))
        self.points = data_utils.generate_point_grid(region, tile_size, self.pixel_w, overlap)

        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        :param idx: index
        :return: dict with 'input' tensor, 'ground truth' tensor and 'coords' (x,y)
        """
        coord = self.points.iloc[idx]

        offset = ((coord.x - self.ulx) / self.pixel_w, (self.uly - coord.y) / self.pixel_w)

        image = data_utils.get_all_bands_as_numpy(self.vrt, offset, self.tile_size, self.bands, self.means, self.stds)
        image = data_utils.redistribute_satellite_data(image)

        # add DEM after changing brightness etc but before rotating and flipping
        if self.dem:
            dem_offset = ((coord.x - self.dem_ulx) / self.pixel_w, (self.dem_uly - coord.y) / self.pixel_w)
            dem_image = data_utils.get_all_bands_as_numpy(self.dem, dem_offset, self.tile_size,
                                                          means=[2100], stds=[1000])
            image = np.concatenate([image, dem_image], axis=2)

        image = self.to_tensor(image)

        out = {'input': image, 'coords': (coord.x, coord.y)}
        if self.aval_path:
            aval_offset = ((coord.x - self.aval_ulx) / self.pixel_w, (self.aval_uly - coord.y) / self.pixel_w)
            gt = data_utils.get_all_bands_as_numpy(self.aval_raster, aval_offset, self.tile_size)
            out['ground truth'] = self.to_tensor(gt)
        return out


if __name__ == '__main__':
    # run test

    # data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
    data_folder = '/home/patrick/ecovision/data/2019'
    ava_file = 'avalanches0119_endversion.shp'
    region_file = 'Multiple_regions.shp'

    my_dataset = AvalancheDatasetGrid(data_folder, ava_file, region_file)
    dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)
    batch = next(dataiter)

    viz_utils.plot_avalanches_by_certainty(*batch)