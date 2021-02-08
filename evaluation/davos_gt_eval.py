""" This script is used to evaluate models with respect to ground truth data different from the avalanche labels,
available around Davos.

It can either be run in visualise difference mode, where one can step through examples were predictions are different to
avalanche labels and save interesting ones. Or, it is run only to compute some statistics with respect to the ground
truth data.

For more information on all flags run the script with the --help flag.
"""

import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from experiments.easy_experiment import EasyExperiment
from datasets.davos_gt_dataset import DavosGtDataset
from torch.utils.data import DataLoader
from utils.utils import str2bool
from utils.viz_utils import viz_predictions, save_fig
from utils.data_augmentation import center_crop_batch


def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('Loading model...')
    model = EasyExperiment.load_from_checkpoint(args.ckpt_path)
    model.eval()

    test_set = DavosGtDataset(args.test_root_dir,
                              args.test_gt_file,
                              args.test_ava_file,
                              dem_path=model.hparams.dem_dir,
                              tile_size=args.tile_size,
                              bands=model.hparams.bands,
                              means=model.hparams.means,
                              stds=model.hparams.stds,
                              )

    test_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=3, drop_last=False, pin_memory=True)

    # Only calculate statistics if not in visual mode
    if not args.viz_diffs:
        mylogger = TensorBoardLogger(args.log_dir, name=args.exp_name)
        trainer = Trainer.from_argparse_args(args, logger=mylogger)
        trainer.test(model, test_loader)
        return

    # Step through examples step for step when in visual mode
    print('Starting evaluation loop')
    for batch in iter(test_loader):
        x, y, mapped, status, id = batch
        y_hat = model(x)

        # aval detected if average in 10px patch around point is bigger than 0.5 threshold
        y_hat_crop = center_crop_batch(y_hat, crop_size=10)
        pred = y_hat_crop.mean(dim=[1, 2, 3]) > 0.5

        # only view samples different from gt
        if (pred == mapped).all():
            continue

        fig = viz_predictions(x, y, y_hat, dem=test_set.dem, gt=status.squeeze())
        fig.show()

        name = input("Enter name to save under or press enter to skip:\n")
        if name:
            print('saving...')
            save_fig(fig, args.save_dir, name)
        else:
            print('searching for next difference...')


if __name__ == "__main__":
    parser = ArgumentParser(description='test and compare avalanche mapping to gt in davos area')

    parser.add_argument('--ckpt_path', type=str, default='best', help='Path to checkpoint to be loaded')
    parser.add_argument('--viz_diffs', type=str2bool, default=False, help='Whether to plot and show samples that are different')
    parser.add_argument('--save_dir', type=str, help='directory under which to save figures')

    # Trainer args
    parser.add_argument('--date', type=str, default='None', help='date when experiment was run')
    parser.add_argument('--time', type=str, default='None', help='time when experiment was run')
    parser.add_argument('--exp_name', type=str, default="default", help='experiment name')
    parser.add_argument('--log_dir', type=str, default=os.getcwd(), help='directory to store logs and checkpoints')

    # Dataset Args
    parser = DavosGtDataset.add_argparse_args(parser)

    # Dataset paths
    parser.add_argument('--test_root_dir', type=str, default='/home/patrick/ecovision/data/2018',
                        help='root directory of the training set')
    parser.add_argument('--test_ava_file', type=str, default='avalanches0118_endversion.shp',
                        help='File name of avalanche shapefile in root directory of training set')
    parser.add_argument('--test_gt_file', type=str, default='Methodenvergleich2018.shp',
                        help='File name of gt comparison data in davos')
    parser.add_argument('--dem_dir', type=str, default=None,
                        help='directory of the DEM within root_dir')

    # Model specific args
    parser = EasyExperiment.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
