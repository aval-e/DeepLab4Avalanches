import os
from osgeo import gdal, osr
import geopandas as gpd
from tqdm import tqdm
from argparse import ArgumentParser
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset_grid import AvalancheDatasetGrid
from torch.utils.data import DataLoader
from utils.losses import crop_to_center
from utils import data_utils


def create_raster(region_file, output_dir, tile_size, pixel_w):
    region = gpd.read_file(region_file)
    minx, miny, maxx, maxy = region.buffer(tile_size, join_style=2).total_bounds
    x_size = int((maxx - minx) // pixel_w)
    y_size = int((maxy - miny) // pixel_w)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2056)

    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(os.path.join(output_dir, 'predictions.tif'), x_size, y_size, 1, gdal.GDT_Float32)
    driver = None
    out_raster.SetGeoTransform((minx, pixel_w, 0, maxy, 0, -pixel_w))
    out_raster.SetProjection(srs.ExportToWkt())
    band_out = out_raster.GetRasterBand(1)
    band_out.SetNoDataValue(0)
    return out_raster


def write_prediction(band, array, coords, border, ulx, uly, pixel_w, tile_size):
    xoff = int((coords[0] - ulx) / pixel_w + border)
    yoff = int((uly - coords[1]) / pixel_w + border)

    band.WriteArray(array.squeeze().cpu().numpy(), xoff=xoff, yoff=yoff)
    band.FlushCache()


def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    model = EasyExperiment.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.freeze()
    model.cuda()

    tile_size = 512
    border = 100
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

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=3,
                            drop_last=False, pin_memory=True)

    pixel_w = test_set.pixel_w
    out_raster = create_raster(args.region_file, args.output_dir, tile_size, pixel_w)
    out_band = out_raster.GetRasterBand(1)
    ulx, uly, _, _ = data_utils.get_raster_extent(out_raster)

    for sample in tqdm(iter(test_loader), desc='Predicting'):
        x = sample['input'].cuda()
        # y_hat = model(x)
        # y_hat = crop_to_center(y_hat, border)
        y_hat = crop_to_center(x[:, [0], :, :], border)

        write_prediction(out_band, y_hat, sample['coords'], border, ulx, uly, pixel_w, tile_size)


if __name__ == "__main__":
    parser = ArgumentParser(description='Run avalanche prediction on satellite images')

    # Trainer args
    parser.add_argument('--image_dir', type=str, default='None', help='directory containing all satellite images')
    parser.add_argument('--dem_path', type=str, default='', help='path to DEM if needed')
    parser.add_argument('--region_file', type=str, default='None', help='path to region file specifying which area to predict')
    parser.add_argument('--output_dir', type=str, default='None', help='directory in which output file is created')
    parser.add_argument('--checkpoint', type=str, default='None', help='model checkpoint to use')
    parser.add_argument('--aval_path', type=str, default='', help='ground truth avalanche path if available for computing metrics')
    args = parser.parse_args()
    main(args)
