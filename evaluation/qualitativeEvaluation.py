""" This script was used for the qualitative evaluation of models.

Predictions are visualised with matplotlib one sample at a time, showing input, ground truth and predictions for all models specified by checkpoints.
Each sample can either be saved to the save_dir by entering a name for it, or be skipped by pressing enter without
entering a name.

A list of checkpoints can be given to directly compare models.
"""

save_dir = '/path/to/location/to/save/illustrations'

checkpoint_dir = '/path/to/the/checkpointsdirectory/'
checkpoints = [checkpoint_dir + 'model1.ckpt',
               checkpoint_dir + 'model2.ckpt',
               checkpoint_dir + 'model3.ckpt'
               ]

dem_dir = '/path/to/DEM/DHM.tif'

import os
import torch
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset_points import AvalancheDatasetPoints
from torch.utils.data import DataLoader, ConcatDataset
from utils.viz_utils import viz_predictions, save_fig
from utils.losses import crop_to_center


def load_test_set(hparams, year='both'):
    """ load the test set for 1st, 2nd or both years"""
    root_dir = '/path/to/root/directory/'
    if year == '18' or year == 'both':
        test_set = AvalancheDatasetPointsEval(root_dir + '2018_wallis',
                                              'avalanches0118_endversion.shp',
                                              'Test_area_2018_TC.shp',
                                              dem_path=hparams.dem_dir,
                                              tile_size=hparams.tile_size,
                                              bands=hparams.bands,
                                              means=hparams.means,
                                              stds=hparams.stds,
                                              )
    if year == '19' or year == 'both':
        test_set2 = AvalancheDatasetPointsEval(root_dir + '2019',
                                               'avalanches0119_endversion.shp',
                                               'Test_area_2019_TC.shp',
                                               dem_path=hparams.dem_dir,
                                               tile_size=hparams.tile_size,
                                               bands=hparams.bands,
                                               means=hparams.means,
                                               stds=hparams.stds,
                                               )
    if year == '19':
        test_set = test_set2
    if year == '18_Mattertal':
        test_set = AvalancheDatasetPointsEval(root_dir + '18_Mattertal',
                                              '20180106_avalanches_MattVdH.shp',
                                              'Test_Mattertal_06012018.shp',
                                              dem_path=hparams.dem_dir,
                                              tile_size=hparams.tile_size,
                                              bands=hparams.bands,
                                              means=hparams.means,
                                              stds=hparams.stds,
                                              )
    elif year == 'both':
        test_set = ConcatDataset([test_set, test_set2]) # concat assembles 1st and 2nd year datasets

    return test_set


def main():
    print("Starting qualitative analysis...")
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
    for batch in iter(val_loader):
        x, y = batch
        x = x.cuda()
        y = y.cuda()

        y_hat = []
        for model in models:
            y_hat.append(model(x)) #model/Easy Experiment is running on input x
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
