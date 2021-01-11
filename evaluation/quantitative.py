import torch
import pandas
import numpy as np
from tqdm import tqdm
from torch.nn import BCELoss
from pytorch_lightning import seed_everything
from experiments.easy_experiment import EasyExperiment
from datasets.avalanche_dataset_points import AvalancheDatasetPointsEval
from torch.utils.data import DataLoader, ConcatDataset
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
                                              tile_size=hparams.tile_size,
                                              bands=hparams.bands,
                                              means=hparams.means,
                                              stds=hparams.stds,
                                        )
    if year == '19' or 'both':
        test_set2 = AvalancheDatasetPointsEval(train_root_dir19,
                                               aval_file19,
                                               region19,
                                               dem_path=hparams.dem_dir,
                                               tile_size=hparams.tile_size,
                                               bands=hparams.bands,
                                               means=hparams.means,
                                               stds=hparams.stds,
                                )
    if year == '19':
        test_set = test_set2
    elif year == 'both':
        test_set = ConcatDataset([test_set, test_set2])

    return test_set


def calc_metrics(soft_metrics, hard_metrics, y_individual, y_hat, thresholds=(0.4, 0.5, 0.6)):
    y_individual = crop_to_center(y_individual)
    y_hat = crop_to_center(y_hat)

    # compress gt avalanches into one image with more certain avalanches on top
    y, _ = y_individual.max(dim=1, keepdim=True)
    y[y > 0] = y_individual[y > 0].min(dim=1, keepdim=True)

    y_mask = data_utils.labels_to_mask(y)  # binary mask ignoring avalanche certainty
    aval_info = per_aval_info(y_hat, y_individual)

    # soft metrics
    soft_metrics['bce'].append(bce(y_hat, y_mask))
    soft_metrics['soft_dice'].append(soft_dice(y_mask, y_hat))
    soft_metrics['soft_recall'].append(aval_info['soft_recall'])

    # hard metrics
    for threshold in thresholds:
        pred = torch.round(y_hat + (0.5 - threshold))  # rounds probability to 0 or 1
        precision, recall, f1 = get_precision_recall_f1(y, pred)
        _, _, f1_no_aval = get_precision_recall_f1(y_mask == 0, pred == 0)
        f1_avg = 0.5 * (f1_no_aval + f1)

        hard_metrics[threshold]['precision'].append(precision)
        hard_metrics[threshold]['recall'].append(recall)
        hard_metrics[threshold]['f1'].append(f1)
        hard_metrics[threshold]['f1_avg'].append(f1_avg)

        # per avalanche metrics
        accuracy = per_aval_accuracy(pred, y_individual)
        for key, val in accuracy.items():
            hard_metrics[threshold][key].extend(val)
        add statistics


    return soft_metrics, hard_metrics


def create_empty_metrics(thresholds, hard_metric_names):
    soft_metrics = {}
    soft_metrics['bce'] = []
    soft_metrics['soft_dice'] = []
    soft_metrics['soft_recall'] = []

    hard_metrics = {}
    for threshold in thresholds:
        thresh_m = {}
        for m in hard_metric_names:
            thresh_m[m] = []
        hard_metrics[threshold] = thresh_m

    return soft_metrics, hard_metrics


def append_avg_metrics_to_dataframe(df, name, metrics):
    soft, hard = metrics

    avg_metrics = {}

    detected = np.array(hard['0.5']['acc_0.7'])
    areas = np.array(hard['0.5']['area_m2'])
    certainties = np.array(hard['0.5']['certainty'])

    # calc statistics
    avg_metrics[('0.5', '0.7_detected_area')] = areas[detected].mean().item()
    avg_metrics[('0.5', '0.7_undetected_area')] = areas[~detected].mean().item()
    avg_metrics[('0.5', '0.7_acc_c1')] = detected[certainties == 1].mean().item()
    avg_metrics[('0.5', '0.7_acc_c2')] = detected[certainties == 2].mean().item()
    avg_metrics[('0.5', '0.7_acc_c3')] = detected[certainties == 3].mean().item()

    # average metrics
    for key, val in soft.items():
        avg_metrics[(key, None)] = np.array(val).mean().item()
    for thresh, hm in hard.items():
        for key, val in hard.items():
            avg_metrics[(thresh, key)] = np.array(val).mean().item()

    df = df.loc[name, :] = avg_metrics
    return df


def main():
    output_path = 'metrics'

    checkpoints = [{'Name': name, 'Year': 'both', 'path': path},
                  ]

    seed_everything(42)

    thresholds = (0.4, 0.5, 0.6)

    # create dataframe to store results
    stats_names = ['0.7_detected_area', '0.7_undetected_area', '0.7_acc_c1', '0.7_acc_c2', '0.7_acc_c3']
    hard_metric_names = ['precision', 'recall', 'F1', 'F1_avg', 'acc_0.5', 'acc_0.7', 'acc_0.8', 'cover_acc']
    myColumns = pandas.MultiIndex.from_tuples([('BCE', None), ('soft_dice', None), ('soft_recall', None),
                                   (thresholds[0], hard_metric_names), (thresholds[1], hard_metric_names.extend(stats_names)),
                                   (thresholds[2], hard_metric_names)])
    myIndex = pandas.Index(data=['Name', 'Year'])
    df = pandas.DataFrame(columns=myColumns, index=myIndex)
    
    with torch.no_grad():
        for checkpoint in checkpoints:
            model = load_model(checkpoint['path'])

            dataset = load_test_set(model.hparams, checkpoint['Year'])
            test_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                                     drop_last=False, pin_memory=True)

            metrics = create_empty_metrics(thresholds, hard_metric_names)

            for batch in tqdm(iter(test_loader), desc='Testing: ' + checkpoint['Name']):
                x, y = batch
                y_hat = model(x)

                # Todo: Check if metrics need to be returned or if appending within function is enough
                calc_metrics(*metrics, y, y_hat, thresholds)

            df = append_avg_metrics_to_dataframe(df, checkpoint['Name'], metrics)
            df.to_csv(output_path + '.csv')

    # export results
    df.to_excel(output_path)


if __name__ == '__main__':
    main()