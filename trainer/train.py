from argparse import ArgumentParser
from pytorch_lightning import Trainer
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset_grid import AvalancheDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def main(hparams):
    my_dataset = AvalancheDataset(hparams.train_root_dir, hparams.train_ava_file, hparams.train_region_file,
                                  transform=ToTensor(), tile_size=hparams.tile_size)
    dataloader = DataLoader(my_dataset, batch_size=hparams.batch_size, shuffle=True, num_workers=1, drop_last=True)

    model = EasyExperiment(hparams)
    trainer = Trainer.from_argparse_args(hparams)

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    parser = ArgumentParser(description='train avalanche detection network')

    # Dataset Args
    parser.add_argument('--train_root_dir', type=str, default='/home/patrick/ecovision/data/2018',
                        help='root directory of the training set')
    parser.add_argument('--train_ava_file', type=str, default='avalanches0118_endversion.shp',
                        help='File name of avalanche shapefile in root directory')
    parser.add_argument('--train_region_file', type=str, default='Region_Selection.shp',
                        help='File name of shapefile in root directory defining training area')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size used in training')
    parser.add_argument('--tile_size', type=tuple, nargs=2, default= (600, 600), help='patch size during training in meters')

    # Trainer args
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    main(hparams)
