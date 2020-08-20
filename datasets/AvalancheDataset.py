from torch.utils.data import Dataset
from utils.data_utils import get_all_bands_as_numpy, get_numpy_from_shapefile
from osgeo import gdal, ogr
import os
import glob

class AvalancheDataset(Dataset):
    """ SLF Avalanche Dataset"""

    def __init__(self, root_dir, image_dirs, shapefile, transform=None):
        self.transform = transform

        # open satellite images
        all_tiffs = []
        for image_dir in image_dirs:
            all_tiffs += glob.glob(os.path.join(os.path.join(root_dir, image_dir), '*.TIF'))
        self.vrt = gdal.BuildVRT('/tmp/myVRT.vrt', all_tiffs)

        # get avalanche shapes
        shape_path = os.path.join(root_dir, shapefile)
        self.shapes = ogr.Open(shape_path)

    def __len__(self):

    def __getitem(self, idx):