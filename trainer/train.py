import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset import AvalancheDataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor


def main(hparams):
    seed_everything(hparams.seed)

    train_set = AvalancheDataset(hparams.train_root_dir,
                                 hparams.train_ava_file,
                                 hparams.train_region_file,
                                 dem_path=hparams.dem_dir,
                                 random=True,
                                 tile_size=hparams.tile_size,
                                 certainty=hparams.aval_certainty,
                                 transform=ToTensor(),
                                 )

    val_set = AvalancheDataset(hparams.val_root_dir,
                               hparams.val_ava_file,
                               hparams.val_region_file,
                               dem_path=hparams.dem_dir,
                               random=False,
                               tile_size=hparams.tile_size,
                               certainty=None,
                               transform=ToTensor(),
                               )

    train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, num_workers=hparams.num_workers,
                              drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers,
                            drop_last=False, pin_memory=True)

    model = EasyExperiment(hparams)
    mylogger = TensorBoardLogger(hparams.log_dir, name=hparams.exp_name)
    trainer = Trainer.from_argparse_args(hparams, logger=mylogger)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = ArgumentParser(description='train avalanche mapping network')

    # Trainer args
    parser.add_argument('--date', type=str, default='None', help='date when experiment was run')
    parser.add_argument('--time', type=str, default='None', help='time when experiment was run')
    parser.add_argument('--exp_name', type=str, default="default", help='experiment name')
    parser.add_argument('--seed', type=int, default=42, help='seed to init all random generators for reproducibility')
    parser.add_argument('--log_dir', type=str, default=os.getcwd(), help='directory to store logs and checkpoints')

    # Dataset Args
    parser.add_argument('--batch_size', type=int, default=2, help='batch size used in training')
    parser.add_argument('--tile_size', type=int, nargs=2, default=[256, 256],
                        help='patch size during training in pixels')
    parser.add_argument('--aval_certainty', type=int, default=None,
                        help='Which avalanche certainty to consider. 1: exact, 2: estimated, 3: guessed')
    parser.add_argument('--num_workers', type=int, default=4, help='no. of workers each dataloader uses')

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

    # Model specific args
    parser = EasyExperiment.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    main(hparams)
