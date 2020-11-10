import os
import geopandas as gpd
from torch.utils.data import DataLoader
import rasterio.features
from shapely.geometry import Point
import shapely.affinity
import affine
from datasets.avalanche_dataset_points import AvalancheDatasetPoints
from utils import data_utils, viz_utils, utils
from torchvision.transforms import ToTensor
from utils.data_augmentation import RandomScaling, RandomShift
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import torch

DEBUG = True
TYP_2_LABEL = {'UNKNOWN': 0,
               'SLAB': 1,
               'LOOSE_SNOW': 2,
               'FULL_DEPTH': 3}


class AvalancheInstDataset(AvalancheDatasetPoints):
    """
    SLF Avalanche Dataset. Similar to AvalancheDatasetPoints but for instance segmentation

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

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        :param idx: index
        :return: [image, rasterised avalanches] as list
        """
        p = self.sample_points.iloc[idx]

        # Get all avalanches in bounding box of possible samples
        bb = self.tile_size * self.pixel_w * 5 / 6
        avals = self.avalanches.cx[p.x - bb:p.x + bb, p.y - bb:p.y + bb]

        # get no of samples according to batch augmentation
        samples = []
        for sample in range(self.ba):
            px_offset = np.array(2 * [self.tile_size]) // 2
            rand_px_offset = np.zeros(2)
            if self.random:
                max_diff = self.tile_size // 3
                rand_px_offset = np.random.randint(-max_diff, max_diff, 2)
                px_offset += rand_px_offset
            vrt_offset = np.array([p.x - self.ulx, self.uly - p.y])
            vrt_offset = vrt_offset / self.pixel_w - px_offset

            image = data_utils.get_all_bands_as_numpy(self.vrt, vrt_offset, self.tile_size,
                                                      means=self.means, stds=self.stds, bands=self.bands)

            # augment brightness and contrast
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

            # -- Apply transforms -----
            angle = self.rand_rotation.get_param()
            image = self.rand_rotation(image, angle)
            image = self.to_tensor(image)
            flip = self.rand_flip.get_param()
            image = self.rand_flip(image, flip)

            # -- Get avalanche instances -----
            rand_offset = rand_px_offset * self.pixel_w
            rand_offset[1] *= -1 # convert from pixel coords to geog. coords
            new_p = Point(np.array([p.x, p.y]) - rand_offset)
            patch_poly = new_p.buffer(self.pixel_w * self.tile_size / 2, cap_style=3)
            bb_patch = patch_poly.bounds
            if DEBUG:
                fig, ax = plt.subplots()
                avals.plot(ax=ax)
                gpd.GeoSeries(shapely.affinity.rotate(patch_poly, -angle, origin=new_p)).boundary.plot(ax=ax,color='red')
                ax.scatter(p.x, p.y, c='red', s=20 ** 2, marker='+')
                plt.show()
                plt.imshow(((image - image.min())/(image.max() - image.min())).permute(1,2,0))
                plt.show()

            masks = np.empty([0, self.tile_size, self.tile_size])
            boxes = np.empty([0, 4])
            labels = np.empty(0)
            for index, aval in avals.iterrows():
                inter = aval.geometry.intersection(patch_poly)
                inter = inter.intersection(shapely.affinity.rotate(patch_poly, -angle, origin=new_p))
                # disregard if visible area is too small
                if inter.area < 400:
                    continue

                # apply same transform as to image
                inter = shapely.affinity.rotate(inter, angle, origin=new_p)
                if flip:
                    inter = shapely.affinity.scale(inter, -1, 1, origin=new_p)

                t = affine.Affine(1.5, 0, bb_patch[0], 0, -1.5, bb_patch[3])
                mask = rasterio.features.rasterize(((inter, 1),), 2 * [self.tile_size], transform=t, dtype=np.single)
                masks = np.append(masks, np.expand_dims(mask, axis=0), axis=0)

                # get bounding box and convert to image coordinates
                bb = ((np.array(inter.bounds) - np.tile(bb_patch[0:2], 2)) / self.pixel_w).round()
                bb = [bb[0], self.tile_size - bb[3], bb[2], self.tile_size - bb[1]]
                boxes = np.append(boxes, np.array(bb, ndmin=2), axis=0)

                labels = np.append(labels, TYP_2_LABEL[aval['typ']])

                if DEBUG:
                    fig, ax = plt.subplots()
                    ax.imshow(mask)
                    rect = patches.Rectangle(bb[0:2], bb[2] - bb[0], bb[3] - bb[1],
                                             edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    fig.show()

            targets = {'boxes': boxes,
                       'labels': labels,
                       'masks': masks}

            samples.append((image, targets))

        return samples if self.ba > 1 else samples[0]


if __name__ == '__main__':
    # run test

    # home
    data_folder = '/home/patrick/ecovision/data/2018'
    ava_file = 'avalanches0118_endversion.shp'
    region_file = 'Small_test_area.shp'
    dem_path = None  # '/home/patrick/ecovision/data/2018/avalanches0118_endversion.tif'

    # pfpc
    # data_folder = '/home/pf/pfstud/bartonp/slf_avalanches/2018'
    # ava_file = 'avalanches0118_endversion.shp'
    # region_file = 'Val_area_2018.shp'
    # dem_path='/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056.tif'
    import random
    random.seed(2)
    np.random.seed(2)
    my_dataset = AvalancheInstDataset(data_folder, ava_file, region_file, tile_size=512, dem_path=dem_path, bands=[1,2,3],
                                      random=True, batch_augm=1, hflip_p=0.5, rand_rot=180)
    dataloader = DataLoader(my_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=utils.inst_collate_fn)

    for batch in iter(dataloader):
        # viz_utils.plot_avalanches_by_certainty(*batch, dem=my_dataset.dem)
        input('Press key for another sample')
