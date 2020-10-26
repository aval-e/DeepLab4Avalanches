import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from experiments.easy_experiment import EasyExperiment
from datasets.davos_gt_dataset import DavosGtDataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor


def main(hparams):
    model = EasyExperiment.load_from_checkpoint(hparams.ckpt_path)
    mylogger = TensorBoardLogger(hparams.log_dir, name=hparams.exp_name)
    trainer = Trainer.from_argparse_args(hparams, logger=mylogger)

    test_set = DavosGtDataset(hparams.test_root_dir,
                              hparams.test_gt_file,
                              hparams.test_ava_file,
                              dem_path=hparams.dem_dir,
                              tile_size=hparams.tile_size,
                              bands=hparams.bands,
                              means=hparams.means,
                              stds=hparams.stds,
                              transform=ToTensor()
                              )

    test_loader = DataLoader(test_set, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers,
                             drop_last=False, pin_memory=True)

    trainer.test(model, test_loader)


if __name__ == "__main__":
    parser = ArgumentParser(description='test and compare avalanche mapping to gt in davos area')

    parser.add_argument('--ckpt_path', type=str, default='best', help='Path to checkpoint to be loaded')

    # Trainer args
    parser.add_argument('--date', type=str, default='None', help='date when experiment was run')
    parser.add_argument('--time', type=str, default='None', help='time when experiment was run')
    parser.add_argument('--exp_name', type=str, default="default", help='experiment name')
    parser.add_argument('--log_dir', type=str, default=os.getcwd(), help='directory to store logs and checkpoints')

    # Dataset Args
    parser.add_argument('--batch_size', type=int, default=2, help='batch size used in training')
    parser.add_argument('--tile_size', type=int, nargs=2, default=[256, 256],
                        help='patch size during training in pixels')
    parser.add_argument('--aval_certainty', type=int, default=None,
                        help='Which avalanche certainty to consider. 1: exact, 2: estimated, 3: guessed')
    parser.add_argument('--bands', type=int, nargs='+', default=None, help='bands from optical imagery to be used')
    parser.add_argument('--means', type=float, nargs='+', default=None,
                        help='list of means to standardise optical images')
    parser.add_argument('--stds', type=float, nargs='+', default=None,
                        help='list of standard deviations to standardise optical images')
    parser.add_argument('--num_workers', type=int, default=4, help='no. of workers each dataloader uses')

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
