import os
from argparse import ArgumentParser
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset_points import AvalancheDatasetPoints
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from utils.viz_utils import viz_predictions, save_fig


def main(hparams):
    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    model = EasyExperiment.load_from_checkpoint(hparams.checkpoint)
    model.eval()

    print('Preparing dataset...')
    val_set = AvalancheDatasetPoints(hparams.val_root_dir,
                                     hparams.val_ava_file,
                                     hparams.val_region_file,
                                     dem_path=hparams.dem_dir,
                                     random=False,
                                     tile_size=hparams.tile_size,
                                     bands=hparams.bands,
                                     certainty=None,
                                     means=hparams.means,
                                     stds=hparams.stds,
                                     )

    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=hparams.num_workers,
                            drop_last=False, pin_memory=True)

    print('Starting loop')
    for batch in iter(val_loader):
        x, y = batch
        y_hat = model(x)

        fig = viz_predictions(x, y, y_hat, dem=val_set.dem)
        fig.show()

        name = input("Enter name to save under or press enter to skip:\n")
        if name:
            print('saving...')
            save_fig(fig, hparams.save_dir, name)
        else:
            print('skipping...')


if __name__ == "__main__":
    parser = ArgumentParser(description='train avalanche mapping network')

    # checkpoint path
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint to be loaded')
    parser.add_argument('--save_dir', type=str, help='directory under which to save figures')

    # Dataset Args
    parser = AvalancheDatasetPoints.add_argparse_args([parser])

    # Dataset paths
    parser.add_argument('--train_root_dir', type=str, default='/home/patrick/ecovision/data/2018',
                        help='root directory of the training set')
    parser.add_argument('--train_ava_file', type=str, default='avalanches0118_endversion.shp',
                        help='File name of avalanche shapefile in root directory of training set')
    parser.add_argument('--train_region_file', type=str, default='Region_Selection.shp',
                        help='File name of shapefile in root directory defining training area')
    parser.add_argument('--val_root_dir', type=str, default='/home/patrick/ecovision/data/2018',
                        help='root directory of the validation set')
    parser.add_argument('--val_ava_file', type=str, default='avalanches0118_endversion.shp',
                        help='File name of avalanche shapefile in root directory of training set')
    parser.add_argument('--val_region_file', type=str, default='Region_Selection.shp',
                        help='File name of shapefile in root directory defining validation area')
    parser.add_argument('--dem_dir', type=str, default=None,
                        help='directory of the DEM within root_dir')

    hparams = parser.parse_args()

    main(hparams)
