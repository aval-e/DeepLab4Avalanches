import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from experiments.easy_experiment import EasyExperiment
from datasets.davos_gt_dataset import DavosGtDataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor
from utils.utils import str2bool
from utils.viz_utils import viz_predictions, save_fig
from utils.data_augmentation import center_crop_batch


def main(hparams):
    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    print('Loading model...')
    model = EasyExperiment.load_from_checkpoint(hparams.ckpt_path)
    model.eval()

    test_set = DavosGtDataset(hparams.test_root_dir,
                              hparams.test_gt_file,
                              hparams.test_ava_file,
                              dem_path=hparams.dem_dir,
                              tile_size=hparams.tile_size,
                              bands=hparams.bands,
                              means=hparams.means,
                              stds=hparams.stds,
                              )

    test_loader = DataLoader(test_set, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers,
                             drop_last=False, pin_memory=True)

    if not hparams.viz_diffs:
        mylogger = TensorBoardLogger(hparams.log_dir, name=hparams.exp_name)
        trainer = Trainer.from_argparse_args(hparams, logger=mylogger)
        trainer.test(model, test_loader)
        return

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
            save_fig(fig, hparams.save_dir, name)
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
    hparams = parser.parse_args()

    main(hparams)
