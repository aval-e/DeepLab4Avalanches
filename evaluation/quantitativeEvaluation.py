""" This script is used for evaluating models on various metrics.

Multiple experiments can be run by adding a dictionary to the list for each experiments, describing its name, year to
be tested on, and path to the model checkpoint.

Soft and hard metrics are computed. Hard metrics refer to those where the probability has to be tresholded to get a
a binary mask, and are computed with respect to multiple thresholds for comparison. Soft metrics are those that can be
computed directly from the predicted probabilities.

Soft metrics: BCE, soft_dice, soft_recall - recall summed directly over the raw rather thresholded probabilities
Hard metrics: precision, recall, f1, f1_background, recall_background, precision_background,
              accuracy at 50% coverage, 70% and 80%. acc_cover is the percentage of area covered per avalanche.
"""

import os
import math
import torch
import pandas
import numpy as np
from tqdm import tqdm
from torch.nn import BCELoss
from pytorch_lightning import seed_everything
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset_points import AvalancheDatasetPointsEval
from torch.utils.data import DataLoader, ConcatDataset
from utils.losses import crop_to_center, get_precision_recall_f1, soft_dice
from utils import data_utils

bce = BCELoss()


def load_model(checkpoint_path):
    model = EasyExperiment.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    model.cuda()
    return model


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


def calc_metrics(soft_metrics, hard_metrics, y_individual, y_hat, thresholds=(0.4, 0.5, 0.6)):
    """ Calculate the metrics for one sample. Hard metrics are evaluated with respect to different thresholds"""
    y_individual = crop_to_center(y_individual)
    y_hat = crop_to_center(y_hat)

    # compress gt avalanches into one image with more certain avalanches on top
    y = y_individual.clone()
    y[y == 0] = 10
    y, _ = y.min(dim=1, keepdim=True)
    y[y == 10] = 0

    y_mask = data_utils.labels_to_mask(y)  # binary mask ignoring avalanche certainty
    aval_info = per_aval_info(y_hat, y_individual)

    # soft metrics
    soft_metrics['bce'].append(bce(y_hat, y_mask).item())
    soft_metrics['soft_dice'].append(soft_dice(y_mask, y_hat).item())
    soft_metrics['soft_recall'].extend(aval_info['soft_recall'])
    soft_metrics['area_m2'].extend(aval_info['area_m2'])
    soft_metrics['certainty'].extend(aval_info['certainty'])

    # hard metrics for each threshold
    for threshold in thresholds:
        pred = torch.round(y_hat + (0.5 - threshold))  # rounds probability to 0 or 1
        precision, recall, f1 = get_precision_recall_f1(y, pred)

        # Background
        precision_back, recall_back, f1_back = get_precision_recall_f1(y_mask == 0, pred == 0)

        hard_metrics[threshold]['precision'].append(precision.item())
        hard_metrics[threshold]['recall'].append(recall.item())
        hard_metrics[threshold]['f1'].append(f1.item())
        hard_metrics[threshold]['precision_back'].append(precision_back.item())
        hard_metrics[threshold]['recall_back'].append(recall_back.item())
        hard_metrics[threshold]['f1_back'].append(f1_back.item())

        #print("tp\ttn\tfn\tfp\tprec\trec\tf1\t")
        #print(tp.item(), tn.item(), fn.item(), fp.item(), precision.item(), recall.item(), f1.item())

        # per avalanche metrics
        accuracy = per_aval_accuracy(pred, y_individual)
        for key, val in accuracy.items():
            hard_metrics[threshold][key].extend(val)

    return soft_metrics, hard_metrics


def per_aval_accuracy(predictions, targets, detection_thresh=(0.5, 0.7, 0.8)):
    """ Accuracy per avalanche. Detection is determined by a minimum area of the avalanche that is predicted as
    avalanche """
    d = {'acc_cover': []}
    thresh_keys = []
    for thresh in detection_thresh:
        key = 'acc_' + str(thresh)
        thresh_keys.append(key)
        d[key] = []

    for i in range(predictions.shape[0]):
        prediction = predictions[i, :, :, :]
        target = targets[i, :, :, :]
        for mask in target:
            mask_sum = (mask > 0).sum().item()
            acc = prediction[:, mask > 0].sum().item() / mask_sum if mask_sum else float('NaN')
            d['acc_cover'].append(acc)
            for i in range(len(detection_thresh)):
                d[thresh_keys[i]].append(acc > detection_thresh[i] if not math.isnan(acc) else float('NaN'))
    return d


def per_aval_info(y_hats, targets):
    """ Some useful information on a per avalanche basis, and soft metrics from predicted probabilities"""
    soft_recall = []
    area = []
    certainty = []
    for i in range(y_hats.shape[0]):
        y_hat = y_hats[i, :, :, :]
        target = targets[i, :, :, :]
        for mask in target:
            masked_pred = y_hat[:, mask > 0]
            size = (mask > 0).sum().item()
            soft_recall.append(masked_pred.sum().item() / size if size else float('NaN'))
            area.append(size * 2.25)  # multiply by 1.5^2 to get meters^2
            certainty.append(mask.max().item())
    return {'soft_recall': soft_recall, 'area_m2': area, 'certainty': certainty}


def create_empty_metrics(thresholds, hard_metric_names):
    """ Initialises the metrics dictionaries with empty lists """
    soft_metrics = {}
    soft_metrics['bce'] = []
    soft_metrics['soft_dice'] = []
    soft_metrics['soft_recall'] = []
    soft_metrics['area_m2'] = []
    soft_metrics['certainty'] = []

    hard_metrics = {}
    for threshold in thresholds:
        thresh_m = {}
        for m in hard_metric_names:
            thresh_m[m] = []
        hard_metrics[threshold] = thresh_m

    return soft_metrics, hard_metrics


def append_avg_metrics_to_dataframe(df, name, year, metrics, columns):
    """ Computes the average metrics across the dataset and appends an entry for that experiment to the
    pandas dataframe """
    soft, hard = metrics

    avg_metrics = {}

    detected = np.array(hard[0.5]['acc_0.7'])
    areas = np.array(soft['area_m2'])
    certainties = np.array(soft['certainty'])

    # calc statistics
    avg_metrics[(0.5, '0.7_detected_area')] = np.nanmean(areas[detected == 1]).item()
    avg_metrics[(0.5, '0.7_undetected_area')] = np.nanmean(areas[detected == 0]).item()
    avg_metrics[(0.5, '0.7_acc_c1')] = np.nanmean(detected[certainties == 1]).item()
    avg_metrics[(0.5, '0.7_acc_c2')] = np.nanmean(detected[certainties == 2]).item()
    avg_metrics[(0.5, '0.7_acc_c3')] = np.nanmean(detected[certainties == 3]).item()

    # average metrics
    for key, val in soft.items():
        avg_metrics[(0, key)] = np.nanmean(np.array(val)).item()
    for thresh, hm in hard.items():
        for key, val in hm.items():
            avg_metrics[(thresh, key)] = np.nanmean(np.array(val)).item()

    s = pandas.Series(data=avg_metrics, index=columns, name=name)
    df.loc[(name, year), :] = s
    return df


def add_all_experiments_in_folder(experiment_folder):
    """ Automatically add all experiments from subfolders. The subfolder is used as the experiment name, and the year
    is set both per default, unless the subfolder name starts with the year it was trained on, in which case it will be
    tested on the other.
    """
    subfolders = [sub for sub in os.listdir(experiment_folder) if os.path.isdir(os.path.join(experiment_folder, sub))]
    experiments = []
    for subfolder in subfolders:
        for dirpath, _, filenames in os.walk(os.path.join(experiment_folder, subfolder)):
            for filename in [f for f in filenames if f.endswith(".ckpt")]:
                if subfolder.startswith('18'):
                    year = '18'
                elif subfolder.startswith('19'):
                    year = '19'
                elif subfolder.startswith('18w'):
                    year = '18'
                    print("subfolder check")
                else:
                    year = 'both'

                experiments.append({'Name': subfolder, 'Year': year,
                                    'path': os.path.join(dirpath, filename)})
    return experiments


def main():
    output_path = '/path/to/output/lightning_logs/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    experiments_folder = '/path/to/experiments/folder'
    # print("experiments folder: ")
    # print(experiments_folder)

    experiments = add_all_experiments_in_folder(experiments_folder)
    # experiments = [{'Name': 'both_deeplabv3+_no_dem', 'Year': 'both',
    #                 'path': experiments_folder + 'both_deeplabv3+_no_dem/version_0/checkpoints/epoch=16.ckpt'},
    #                ]
    # print(experiments)
    seed_everything(42)

    thresholds = (0.4, 0.5)

    # create dataframe with all relevant entries to store results
    stats_names = ['0.7_detected_area', '0.7_undetected_area', '0.7_acc_c1', '0.7_acc_c2', '0.7_acc_c3']
    hard_metric_names = ['precision', 'recall', 'f1', 'f1_back', 'recall_back', 'precision_back', 'acc_0.5', 'acc_0.7',
                         'acc_0.8', 'acc_cover']
    hard_metric_and_stat_names = hard_metric_names.copy()
    hard_metric_and_stat_names.extend(stats_names)
    index_tuples = [(0, 'bce'), (0, 'soft_dice'), (0, 'soft_recall')]
    for thresh in thresholds:
        index_tuples.extend([(thresh, name) for name in hard_metric_names])
    index_tuples.extend([(0.5, name) for name in stats_names])
    myColumns = pandas.MultiIndex.from_tuples(index_tuples)
    myIndex = pandas.MultiIndex.from_tuples([('Name', 'Year')])
    df = pandas.DataFrame(columns=myColumns, index=myIndex)
    print("starting loop over experiments")
    with torch.no_grad():
        # Loop over all experiments
        for experiment in experiments:
            model = load_model(experiment['path'])

            dataset = load_test_set(model.hparams, experiment['Year'])
            test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4,
                                     drop_last=False, pin_memory=True)

            metrics = create_empty_metrics(thresholds, hard_metric_names)

            for j, batch in enumerate(tqdm(iter(test_loader), desc='Testing: ' + experiment['Name'])):
                x, y = batch
                x = x.cuda()
                y = y.cuda()
                y_hat = model(x)

                calc_metrics(*metrics, y, y_hat, thresholds)

            df = append_avg_metrics_to_dataframe(df, experiment['Name'], experiment['Year'], metrics, myColumns)
            #print("Metrics:")
            #print(metrics)

            # save backup in case an error occours later on
            df.to_excel(output_path + 'metrics_backup.xlsx')

    # export results
    df.to_excel(output_path + 'metrics.xlsx')


if __name__ == '__main__':
    main()
