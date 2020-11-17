import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from experiments.easy_experiment import EasyExperiment
from experiments.inst_segm import InstSegmentation
from experiments.detectron_segm import DetectronSegmentation
from datasets.avalanche_dataset_points import AvalancheDatasetPoints
from datasets.avalanche_inst_dataset import AvalancheInstDataset
from datasets.davos_gt_dataset import DavosGtDataset
from datasets.detectron2_dataset import Detectron2Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from utils.utils import str2bool, ba_collate_fn, inst_collate_fn, detectron_collate_fn


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

    # change some params when using Instance segmentation
    my_experiment = EasyExperiment
    my_dataset = AvalancheDatasetPoints
    my_collate_fn = ba_collate_fn
    if hparams.model == 'mask_rcnn':
        my_experiment = InstSegmentation
        my_dataset = AvalancheInstDataset
        my_collate_fn = inst_collate_fn
    elif hparams.model == 'centermask':
        my_experiment = DetectronSegmentation
        my_dataset = Detectron2Dataset
        my_collate_fn = detectron_collate_fn

    # load model
    if hparams.checkpoint:
        if hparams.resume_training:
            model = my_experiment(hparams)
            resume_ckpt = hparams.checkpoint
        else:
            model = my_experiment.load_from_checkpoint(hparams.checkpoint, hparams=hparams)
            resume_ckpt = None
    else:
        model = my_experiment(hparams)
        resume_ckpt = None

    mylogger = TensorBoardLogger(hparams.log_dir, name=hparams.exp_name, default_hp_metric=False)
    mycheckpoint = ModelCheckpoint(monitor='f1/a_soft_dice', mode='max')
    trainer = Trainer.from_argparse_args(hparams, logger=mylogger, checkpoint_callback=mycheckpoint,
                                         resume_from_checkpoint=resume_ckpt,
                                         callbacks=[LearningRateMonitor('step')])

    train_set = my_dataset(hparams.train_root_dir,
                           hparams.train_ava_file,
                           hparams.train_region_file,
                           dem_path=hparams.dem_dir,
                           tile_size=hparams.tile_size,
                           bands=hparams.bands,
                           certainty=hparams.aval_certainty,
                           batch_augm=hparams.batch_augm,
                           means=hparams.means,
                           stds=hparams.stds,
                           random=True,
                           hflip_p=hparams.hflip_p,
                           rand_rot=hparams.rand_rotation,
                           )

    val_set = my_dataset(hparams.val_root_dir,
                         hparams.val_ava_file,
                         hparams.val_region_file,
                         dem_path=hparams.dem_dir,
                         tile_size=512,
                         bands=hparams.bands,
                         certainty=None,
                         batch_augm=0,
                         means=hparams.means,
                         stds=hparams.stds,
                         random=False,
                         hflip_p=0,
                         rand_rot=0,
                         )
    loader_batch_size = hparams.batch_size // hparams.batch_augm if hparams.batch_augm > 0 else hparams.batch_size
    train_loader = DataLoader(train_set, batch_size=loader_batch_size, shuffle=True, num_workers=hparams.num_workers,
                              drop_last=True, pin_memory=True, collate_fn=my_collate_fn)
    val_loader = DataLoader(val_set, batch_size=hparams.batch_size, shuffle=False, num_workers=hparams.num_workers,
                            drop_last=False, pin_memory=True, collate_fn=my_collate_fn)

    trainer.fit(model, train_loader, val_loader)

    # Test and compare on davos ground truth data
    test_set = DavosGtDataset(hparams.val_root_dir,
                              hparams.val_gt_file,
                              hparams.val_ava_file,
                              dem_path=hparams.dem_dir,
                              tile_size=256,
                              bands=hparams.bands,
                              means=hparams.means,
                              stds=hparams.stds,
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
    parser = AvalancheDatasetPoints.add_argparse_args(parser)

    # Dataset paths
    parser.add_argument('--train_root_dir', type=str, default='/home/patrick/ecovision/data/2018',
                        help='root directory of the training set')
    parser.add_argument('--train_ava_file', type=str, default='avalanches0118_endversion.shp',
                        help='File name of avalanche shapefile in root directory of training set')
    parser.add_argument('--train_region_file', type=str, default='Small_test_area.shp',
                        help='File name of shapefile in root directory defining training area')
    parser.add_argument('--val_root_dir', type=str, default='/home/patrick/ecovision/data/2018',
                        help='root directory of the validation set')
    parser.add_argument('--val_ava_file', type=str, default='avalanches0118_endversion.shp',
                        help='File name of avalanche shapefile in root directory of training set')
    parser.add_argument('--val_region_file', type=str, default='Small_test_area.shp',
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
