from argparse import ArgumentParser
from pytorch_lightning import Trainer
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset import AvalancheDataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor


def main(hparams):
    train_set = AvalancheDataset(hparams.train_root_dir,
                                 hparams.train_image_dir,
                                 hparams.train_ava_file,
                                 hparams.train_region_file,
                                 dem_dir=hparams.dem_dir,
                                 random=True,
                                 tile_size=hparams.tile_size,
                                 certainty=hparams.aval_certainty,
                                 transform=ToTensor(),
                                 )

    val_set = AvalancheDataset(hparams.val_root_dir,
                               hparams.val_image_dir,
                               hparams.val_ava_file,
                               hparams.val_region_file,
                               dem_dir=hparams.dem_dir,
                               random=False,
                               tile_size=hparams.tile_size,
                               certainty=hparams.aval_certainty,
                               transform=ToTensor(),
                               )

    train_loader = DataLoader(train_set, batch_size=hparams.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=hparams.batch_size, shuffle=False, num_workers=2, drop_last=False)

    model = EasyExperiment(hparams)
    trainer = Trainer.from_argparse_args(hparams)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = ArgumentParser(description='train avalanche mapping network')

    # Dataset Args
    parser.add_argument('--train_root_dir', type=str, default='/home/patrick/ecovision/data/2018',
                        help='root directory of the training set')
    parser.add_argument('--train_image_dir', type=str, default='orthos_1-5m_RGBN_LV95_16bit_tiff',
                        help='directory within train_root_dir containing satellite images')
    parser.add_argument('--train_ava_file', type=str, default='avalanches0118_endversion.shp',
                        help='File name of avalanche shapefile in root directory of training set')
    parser.add_argument('--train_region_file', type=str, default='Region_Selection.shp',
                        help='File name of shapefile in root directory defining training area')
    parser.add_argument('--val_root_dir', type=str, default='/home/patrick/ecovision/data/2018',
                        help='root directory of the validation set')
    parser.add_argument('--val_image_dir', type=str, default='orthos_1-5m_RGBN_LV95_16bit_tiff',
                        help='directory within val_root_dir containing satellite images')
    parser.add_argument('--val_ava_file', type=str, default='avalanches0118_endversion.shp',
                        help='File name of avalanche shapefile in root directory of training set')
    parser.add_argument('--val_region_file', type=str, default='Region_Selection.shp',
                        help='File name of shapefile in root directory defining validation area')
    parser.add_argument('--dem_dir', type=str, default=None,
                        help='directory of the DEM within root_dir')

    parser.add_argument('--batch_size', type=int, default=2, help='batch size used in training')
    parser.add_argument('--tile_size', type=tuple, nargs=2, default=(256, 256),
                        help='patch size during training in pixels')
    parser.add_argument('--aval_certainty', type=int, default=None,
                        help='Which avalanche certainty to consider. 1: exact, 2: estimated, 3: guessed')

    # Model specific args
    parser = EasyExperiment.add_model_specific_args(parser)

    # Trainer args
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    main(hparams)
