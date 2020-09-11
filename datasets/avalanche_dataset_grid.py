import os
import geopandas as gpd
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal, ogr
from utils import data_utils, viz_utils


class AvalancheDatasetGrid(Dataset):
    """
    SLF Avalanche Dataset. Samples are taken on a grid within region file

    :param root_dir: directory in which all data is located
    :param shape_file: shapefile name located in root_dir of the avalanches
    :param region_file: shapefile containing polygon specifying which area will be considered by the Dataset
    :param tile_size: size in meters (x,y) of a sample
    :param overlap: overlap in meters (x,y) of samples. Use this when context information is important like when only
                    start or end of avalanche is visible.
    :param transform: transform to apply to data. Eg. rotation, toTensor, etc.
    :return pytorch dataset to be used with dataloader
    """

    def __init__(self, root_dir, shape_file, region_file, tile_size=(1000, 1000), overlap=(0, 0),
                 transform=None):
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform

        # open satellite images - all tiffs found in root directory
        all_tiffs = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in [f for f in filenames if f.endswith(".tif") or f.endswith(".TIF")]:
                all_tiffs.append(os.path.join(dirpath, filename))
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
        shp_image = data_utils.get_numpy_from_ogr_shapefile(self.shapes, self.vrt, offset, self.res)
        # image = image[:, :, 0:3]

        if self.transform:
            image = self.transform(image)
            shp_image = self.transform(shp_image)

        return [image, shp_image]


if __name__ == '__main__':
    # run test

    # data_folder = '/media/patrick/Seagate Expansion Drive/SLF_Avaldata/2019'
    data_folder = '/home/patrick/ecovision/data/2019'
    ava_file = 'avalanches0119_endversion.shp'
    # region_file = 'Region_Selection.shp'
    region_file = 'Multiple_regions.shp'

    my_dataset = AvalancheDataset(data_folder, ava_file, region_file)
    dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True, num_workers=1)

    dataiter = iter(dataloader)
    batch = next(dataiter)

    batch = [elem.squeeze() for elem in batch]
    viz_utils.viz_sample(batch)