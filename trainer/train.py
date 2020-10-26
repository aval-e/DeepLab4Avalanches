import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset import AvalancheDataset
from datasets.avalanche_dataset_points import AvalancheDatasetPoints
from datasets.davos_gt_dataset import DavosGtDataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip
from utils.data_augmentation import RandomRotation
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from utils.utils import str2bool, ba_collate_fn


class RunValidationOnStart(Callback):
    """ Run complete evaluation step to check metrics before training
    Todo: Checkpointing does not work anymore when used. Probably registering 0 loss here which is never improved upon
    """

    def __init__(self):
        pass

    def on_train_start(self, trainer: Trainer, pl_module):
        ret_val = trainer.run_evaluation(test_mode=False)
        return ret_val


def main(hparams):
    seed_everything(hparams.seed)

    # load model
    if hparams.checkpoint:
        if hparams.resume_training:
            model = EasyExperiment(hparams)
            resume_ckpt = hparams.checkpoint
        else:
            model = EasyExperiment.load_from_checkpoint(hparams.checkpoint, hparams=hparams)
            resume_ckpt = None
    else:
        model = EasyExperiment(hparams)
        resume_ckpt = None

    mylogger = TensorBoardLogger(hparams.log_dir, name=hparams.exp_name, default_hp_metric=False)
    mycheckpoint = ModelCheckpoint(monitor='f1/a_soft_dice', mode='max')
    trainer = Trainer.from_argparse_args(hparams, logger=mylogger, checkpoint_callback=mycheckpoint,
                                         resume_from_checkpoint=resume_ckpt,
                                         callbacks=[LearningRateMonitor('step')])

    # build transform list since some transforms can only be applied to numpy arrays or torch tensors
    transform_list = []
    if hparams.rand_rotation != 0:
        transform_list.append(RandomRotation(hparams.rand_rotation))
    transform_list.append(ToTensor())
    if hparams.hflip_p != 0:
        transform_list.append(RandomHorizontalFlip(hparams.hflip_p))

    train_set = AvalancheDataset(hparams.train_root_dir,
                                       hparams.train_ava_file,
                                       hparams.train_region_file,
                                       dem_path=hparams.dem_dir,
                                       random=True,
                                       tile_size=hparams.tile_size,
                                       bands=hparams.bands,
                                       certainty=hparams.aval_certainty,
                                       #batch_augm=hparams.batch_augm,
                                       means=hparams.means,
                                       stds=hparams.stds,
                                       transform=Compose(transform_list)
                                       )

    val_set = AvalancheDatasetPoints(hparams.val_root_dir,
                                     hparams.val_ava_file,
                                     hparams.val_region_file,
                                     dem_path=hparams.dem_dir,
                                     random=False,
                                     tile_size=[512, 512],
                                     bands=hparams.bands,
                                     certainty=None,
                                     #batch_augm=0,
                                     means=hparams.means,
                                     stds=hparams.stds,
                                     transform=ToTensor(),
                                     )
    loader_batch_size = hparams.batch_size // hparams.batch_augm if hparams.batch_augm > 0 else hparams.batch_size
    train_loader = DataLoader(train_set, batch_size=loader_batch_size, shuffle=True, num_workers=hparams.num_workers,
                              drop_last=True, pin_memory=True, collate_fn=ba_collate_fn)
    val_loader = DataLoader(val_set, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers,
                            drop_last=False, pin_memory=True, collate_fn=ba_collate_fn)

    trainer.fit(model, train_loader, val_loader)

    # Test and compare on davos ground truth data
    test_set = DavosGtDataset(hparams.val_root_dir,
                              hparams.val_gt_file,
                              hparams.val_ava_file,
                              dem_path=hparams.dem_dir,
                              tile_size=[256, 256],
                              bands=hparams.bands,
                              means=hparams.means,
                              stds=hparams.stds,
                              transform=ToTensor()
                              )
    test_loader = DataLoader(test_set, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers,
                             drop_last=False, pin_memory=True)
    trainer.test(test_dataloaders=test_loader)


if __name__ == "__main__":
    parser = ArgumentParser(description='train avalanche mapping network')

    # Trainer args
    parser.add_argument('--date', type=str, default='None', help='date when experiment was run')
    parser.add_argument('--time', type=str, default='None', help='time when experiment was run')
    parser.add_argument('--exp_name', type=str, default="default", help='experiment name')
    parser.add_argument('--seed', type=int, default=42, help='seed to init all random generators for reproducibility')
    parser.add_argument('--log_dir', type=str, default=os.getcwd(), help='directory to store logs and checkpoints')
    parser.add_argument('--checkpoint', type=str, default='', help='path to checkpoint if one is to be used')
    parser.add_argument('--resume_training', type=str2bool, default=False,
                        help='whether to resume training or only load model weights from checkpoint')

    # Dataset Args
    parser.add_argument('--batch_size', type=int, default=2, help='batch size used in training')
    parser.add_argument('--batch_augm', type=int, default=0, help='the amount of batch augmentation to use')
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

    # data augmentation
    parser.add_argument('--hflip_p', type=float, default=0, help='probability of horizontal flip')
    parser.add_argument('--rand_rotation', type=float, default=0, help='max random rotation in degrees')

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
    parser.add_argument('--val_gt_file', type=str, default='Methodenvergleich2018.shp',
                        help='File name of gt comparison data in davos')

    # Model specific args
    parser = EasyExperiment.add_model_specific_args(parser)

    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    main(hparams)
