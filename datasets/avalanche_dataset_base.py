import os
from argparse import ArgumentParser
from torch.utils.data import Dataset
from utils import data_utils
from osgeo import gdal


class AvalancheDatasetBase(Dataset):
    """
    Base for avalanche datasets since they all open the same data but return different data when getting items

    :param root_dir: directory in which all data is located
    :param aval_file: shapefile name located in root_dir of the avalanches
    :param dem_path: file path of digital elevation model if it is to be used. Default: None
    :param tile_size: patch size to use for training
    :param bands: list of band indexes to read from optical images. Default None gets all
    :param means: list of means for each band in the optical imagery used for standardisation
    :param stds: list of standard deviations for each band in the optical imagery for standardisation
    """

    def __init__(self, root_dir, aval_file, dem_path=None, tile_size=(256, 256), bands=None, means=None, stds=None):
        print('Creating Avalanche Dataset...')
        self.tile_size = tile_size
        self.bands = bands
        self.means = means
        self.stds = stds

        gdal.SetCacheMax(134217728) # 134Mb to limit memory usage on leonhard

        aval_raster_path = os.path.join(root_dir, os.path.splitext(aval_file)[0] + '.tif')
        vrt_padding = 1.5 * self.tile_size # padding around vrts [m] to avoid index error when reading near edge

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

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    @staticmethod
    def add_argparse_args(parent_parser):
        """ add dataset hparams to argparse """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch_size', type=int, default=2, help='batch size used in training')
        parser.add_argument('--tile_size', type=int, default=256, help='patch size during training in pixels')
        parser.add_argument('--bands', type=int, nargs='+', default=None, help='bands from optical imagery to be used')
        parser.add_argument('--means', type=float, nargs='+', default=None,
                            help='list of means to standardise optical images')
        parser.add_argument('--stds', type=float, nargs='+', default=None,
                            help='list of standard deviations to standardise optical images')
        parser.add_argument('--num_workers', type=int, default=4, help='no. of workers each dataloader uses')
        return parser
