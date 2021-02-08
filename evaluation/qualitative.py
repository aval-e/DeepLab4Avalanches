""" This script was used for the qualitative evaluation of models.

Predictions are visualised with matplotlib one sample at a time, showing input, ground truth and predictions.
Each sample can either be saved to the save_dir by entering a name for it, or be skipped by pressing enter without
entering a name.

A list of checkpoints can be given to directly compare models.
"""

save_dir = '/scratch/bartonp/images/presentation'

checkpoint_dir = '/home/pf/pfstud/bartonp/checkpoints/'
checkpoints = [checkpoint_dir + 'both_deeplabv3+/version_0/checkpoints/epoch=16.ckpt',
               checkpoint_dir + 'both_myresnet34/version_0/checkpoints/epoch=16.ckpt'
               ]

dem_dir = '/home/pf/pfstud/bartonp/dem_ch/swissalti3d_2017_ESPG2056_packbits_tiled.tif'

import os
import torch
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset_points import AvalancheDatasetPoints
from torch.utils.data import DataLoader, ConcatDataset
from utils.viz_utils import viz_predictions, save_fig
from utils.losses import crop_to_center


def load_test_set(hparams, year='both'):
    root_dir = '/home/pf/pfstud/bartonp/slf_avalanches/'
    if year == '18' or year == 'both':
        test_set = AvalancheDatasetPoints(root_dir + '2018',
                                          'avalanches0118_endversion.shp',
                                          'Test_area_2018.shp',
                                          dem_path=dem_dir,
                                          random=False,
                                          tile_size=hparams.tile_size,
                                          bands=hparams.bands,
                                          certainty=None,
                                          means=hparams.means,
                                          stds=hparams.stds,
                                          )
    if year == '19' or year == 'both':
        test_set2 = AvalancheDatasetPoints(root_dir + '2019',
                                           'avalanches0119_endversion.shp',
                                           'Test_area_2019.shp',
                                           dem_path=dem_dir,
                                           random=False,
                                           tile_size=hparams.tile_size,
                                           bands=hparams.bands,
                                           certainty=None,
                                           means=hparams.means,
                                           stds=hparams.stds,
                                           )
    if year == '19':
        test_set = test_set2
    elif year == 'both':
        test_set = ConcatDataset([test_set, test_set2])

    return test_set


def main():
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    models = []
    for checkpoint in checkpoints:
        model = EasyExperiment.load_from_checkpoint(checkpoint)
        model.eval()
        model.freeze()
        model.cuda()
        models.append(model)

    test_set = load_test_set(models[0].hparams, year='both')

    val_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=6,
                            drop_last=False, pin_memory=True)

    print('Starting loop')
    for batch in iter(val_loader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        y_hat = []
        for model in models:
            y_hat.append(model(x))
        x = torch.cat(len(y_hat) * [x], dim=0)
        y = torch.cat(len(y_hat) * [y], dim=0)
        y_hat = torch.cat(y_hat, dim=0)

        # x = crop_to_center(x)
        # y = crop_to_center(y)
        # y_hat = crop_to_center(y_hat)
        
        pred = (y_hat + 0.1).round()

        fig = viz_predictions(x, y, y_hat, pred, dem=models[0].hparams.dem_dir, fig_size=4, transpose=True)
        fig.show()

        name = input("Enter name to save under or press enter to skip:\n")
        if name:
            print('saving...')
            save_fig(fig, save_dir, name)
        else:
            print('skipping...')


if __name__ == "__main__":
    main()
