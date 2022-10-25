""" This script can be used to run a model across an entire region and save predictions to a GeoTiff.

To see all configuration options, run the script with the --help flag.
The script can be run having set all relevant flags of paths to the input images, output directory and region file. The
region file is needed to define which area needs to be predicted and is expected to be a shapefile.

Predictions will be written to a GeoTiff stored under the output directory specified with the flags. Predictions are
not thresholded but raw floats between 0 and 1 representing the probability of an avalanche.

A ground truth avalanche file may also be included with the --aval_path flag if metrics are to be computed.
"""

import os
import numpy as np
from osgeo import gdal, osr
import geopandas as gpd
from tqdm import tqdm
from argparse import ArgumentParser
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset_grid import AvalancheDatasetGrid
from torch.utils.data import DataLoader
from utils.losses import crop_to_center, get_precision_recall_f1, soft_dice
from utils import data_utils
import errno


def create_raster(region_file, output_path, tile_size, pixel_w):
    """ Create the output raster such that it covers the entire region to be predicted and uses EPSG 2056 coordinate
    system.
    """
    region = gpd.read_file(region_file)
    minx, miny, maxx, maxy = region.buffer(tile_size, join_style=2).total_bounds
    x_size = int((maxx - minx) // pixel_w)
    y_size = int((maxy - miny) // pixel_w)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2056)

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(os.path.join(output_path), x_size, y_size, 1, gdal.GDT_Float32)
    driver = None
    out_raster.SetGeoTransform((minx, pixel_w, 0, maxy, 0, -pixel_w))
    out_raster.SetProjection(srs.ExportToWkt())
    band_out = out_raster.GetRasterBand(1)
    band_out.SetNoDataValue(0)
    return out_raster


def write_prediction(band, array, coords, border, ulx, uly, pixel_w):
    """ Write the predictions of a patch to the raster.
    :param band: band of the output raster to be written to
    :param array: numpy array of predictions to be written to raster
    :param coords: tuple of (x,y) coordinates of the top left corner of the patch
    :param ulx: coordinate of the most left pixel in the raster
    :param uly: coordinate of the top most pixel in the raster
    :param pixel_w: Pixel width or spatial resolution
    """
    xoff = int((coords[0] - ulx) / pixel_w + border)
    yoff = int((uly - coords[1]) / pixel_w + border)

    band.WriteArray(array.squeeze().cpu().numpy(), xoff=xoff, yoff=yoff)
    band.FlushCache()  # Write cached raster to file


def compute_metrics(metrics, y, y_hat):
    """ Compute metrics for a patch """
    y_mask = data_utils.labels_to_mask(y)
    pred = y_hat.round()

    precision, recall, f1 = get_precision_recall_f1(y, pred)
    precision_back, recall_back, f1_back = get_precision_recall_f1(y_mask == 0, pred == 0)
    metrics['dice_score'].append(soft_dice(y_mask, y_hat).item())
    metrics['precision'].append(precision.item())
    metrics['recall'].append(recall.item())
    metrics['f1'].append(f1.item())
    metrics['precision_back'].append(precision_back.item())
    metrics['recall_back'].append(recall_back.item())
    metrics['f1_back'].append(f1_back.item())


def print_metrics(metrics):
    for key, val in metrics.items():
        metrics[key] = np.nanmean(np.array(val)).item()
    print(metrics)


def init_metrics():
    metrics = {'dice_score': [],
               'precision': [],
               'recall': [],
               'f1': [],
               'precision_back': [],
               'recall_back': [],
               'f1_back': []}
    return metrics


def main(args):
    model = EasyExperiment.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.freeze()
    model.cuda()

    tile_size = args.tile_size
    border = args.border
    test_set = AvalancheDatasetGrid(root_dir=args.image_dir,
                                    region_file=args.region_file,
                                    dem_path=args.dem_path,
                                    aval_path=args.aval_path,
                                    tile_size=tile_size,
                                    overlap=border*2,
                                    bands=[3, 4],
                                    means=[1023.9, 949.9],
                                    stds=[823.4, 975.5],
                                    )

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4,
                            drop_last=False, pin_memory=True)

    pixel_w = test_set.pixel_w
    out_raster = create_raster(args.region_file, args.output_path, tile_size, pixel_w)
    out_band = out_raster.GetRasterBand(1)
    ulx, uly, _, _ = data_utils.get_raster_extent(out_raster)

    if args.aval_path:
        metrics = init_metrics()

    for sample in tqdm(iter(test_loader), desc='Predicting'):
        x = sample['input'].cuda()
        y_hat = model(x)
        y_hat = crop_to_center(y_hat, border)

        write_prediction(out_band, y_hat, sample['coords'], border, ulx, uly, pixel_w)

        if test_set.aval_path:
            y = sample['ground truth'].cuda()
            y = crop_to_center(y[:, [0], :, :], border)
            compute_metrics(metrics, y, y_hat)

    if args.aval_path:
        print('Finished. Computing metrics:')
        print_metrics(metrics)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), string)


if __name__ == "__main__":
    parser = ArgumentParser(description='Run avalanche prediction on satellite images')

    # Trainer args
    parser.add_argument('--image_dir', type=dir_path, required = True, help='directory containing all satellite images')
    parser.add_argument('--dem_path', type=file_path, required = True, default='', help='path to DEM if needed')
    parser.add_argument('--region_file', type=file_path, required = True, help='path to region file specifying which area to predict')
    parser.add_argument('--output_path', type=dir_path, required = True, help='path to output file of predictions. Will be created or overwritten.')
    parser.add_argument('--checkpoint', type=file_path, required = True, help='model checkpoint to use')
    parser.add_argument('--aval_path', type=str, default='', help='ground truth avalanche path if available for computing metrics')
    parser.add_argument('--tile_size', type=int, default=1024, help='Tile size to be used for predictions. Default: 1024')
    parser.add_argument('--border', type=int, default=100, help='Border to be disregarded for each sample in pixels. Default: 100')
    args = parser.parse_args()
    main(args)
