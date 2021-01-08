import os
import torch
from torch.nn import BCELoss
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from experiments.easy_experiment import EasyExperiment
from datasets.davos_gt_dataset import DavosGtDataset
from datasets.avalanche_dataset_points import AvalancheDatasetPointsEval
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.transforms import ToTensor
from utils.utils import str2bool
from utils.viz_utils import viz_predictions, save_fig
from utils.losses import crop_to_center, get_precision_recall_f1, soft_dice, per_aval_accuracy, per_aval_info
from utils import data_utils


bce = BCELoss()


def load_model(checkpoint):
    model = EasyExperiment.load_from_checkpoint(checkpoint)
    model.eval()
    model.freeze()
    model.cuda()
    return model


def load_test_set(hparams, year='both'):
    if year == '18' or year =='both':
        test_set = AvalancheDatasetPointsEval(root_dir18,
                                              aval_file18,
                                              region_file18,
                                              dem_path=hparams.dem_dir,
                                              tile_size=512,
                                              bands=hparams.bands,
                                              means=hparams.means,
                                              stds=hparams.stds,
                                        )
    if year == '19' or 'both':
        test_set2 = AvalancheDatasetPointsEval(train_root_dir19,
                                               aval_file19,
                                               region19,
                                               dem_path=hparams.dem_dir,
                                               tile_size=512,
                                               bands=hparams.bands,
                                               means=hparams.means,
                                               stds=hparams.stds,
                                )
    if year == '19':
        test_set = test_set2
    elif year == 'both':
        test_set = ConcatDataset([test_set, test_set2])

    return test_set


def calc_metrics(y_individual, y_hat, thresholds=(0.4, 0.5, 0.6)):
    y_individual = crop_to_center(y_individual)
    y_hat = crop_to_center(y_hat)

    # compress gt avalanches into one image with more certain avalanches on top
    y, _ = y_individual.max(dim=1, keepdim=True)
    y[y > 0] = y_individual[y > 0].min(dim=1, keepdim=True)

    y_mask = data_utils.labels_to_mask(y)  # binary mask ignoring avalanche certainty
    aval_info = per_aval_info(y_hat, y_individual)

    # soft metrics
    metrics = {}
    metrics['bce'] = bce(y_hat, y_mask)
    metrics['soft_dice'] = soft_dice(y_mask, y_hat)
    metrics['soft_recall'] = aval_info['soft_recall']

    # hard metrics
    for threshold in thresholds:
        pred = torch.round(y_hat + (0.5 - threshold))  # rounds probability to 0 or 1
        precision, recall, f1 = get_precision_recall_f1(y, pred)
        _, _, f1_no_aval = get_precision_recall_f1(y_mask == 0, pred == 0)
        f1_avg = 0.5 * (f1_no_aval + f1)

        metrics[str(threshold) + '_precision'] = precision
        metrics[str(threshold) + '_recall'] = recall
        metrics[str(threshold) + '_f1'] = f1
        metrics[str(threshold) + '_f1_avg'] = f1_avg

        # per avalanche metrics
        accuracy = per_aval_accuracy(pred, y_individual)
        for key, val in accuracy.items():
            metrics[str(threshold) + '_' + key] = val
        add statistics

    return metrics

def save_metrics(file, metrics):


def main():

    checkpoints = {'name': path,

                    }

    metrics file

    
    with torch.no_grad():
        for name, ckpt_path in checkpoints.items():
            model = load_model(ckpt_path)

            dataset = load_test_set(model.hparams, year)
            test_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                                     drop_last=False, pin_memory=True)
            for batch in test_loader:
                x, y = batch
                y_hat = model(x)

                metrics = calc_metrics(y, y_hat)
                append metrics

            dump metrics in csv file

            methoden vergleich


if __name__ == '__main__':
    main()