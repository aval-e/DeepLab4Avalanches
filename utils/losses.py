import torch
import math
from torch.nn import functional as F


def calc_pred_stats(gt, pred):
    """ Calculate the true positive, true negative, false positive and false negative ratios.
        Only a binary mask is considered - zero and nonzero

        :param gt: ground truth values
        :param pred: predicated values
        :return: list of [tp, tn, fp, fn]
    """
    t = pred == (gt != 0)
    f = ~t
    n = float(gt.numel())
    tp = torch.sum(pred[t]) / n
    tn = torch.sum(t) / n - tp
    fp = torch.sum(pred[f]) / n
    fn = torch.sum(f) / n - fp
    return tp, tn, fp, fn


def recall_for_label(gt, pred, label):
    """ Get the recall for specific label or avalanche certainty only
        :param gt: ground truth labels
        :param pred: prediction mask - consist only of 0s and 1s
        :return: recall - fraction of label that was was predicted as 1
    """
    mask = gt == label
    masked_pred = pred[mask]
    n = float(masked_pred.numel())
    tp = torch.sum(masked_pred == 1)
    return tp / n


def precision(tp, fp):
    """ Calculate the precision
        :param tp: true positive ratio
        :param fp: false positive ratio
    """
    return tp / (tp + fp)


def recall(tp, fn):
    """ Calculate the recall
        :param tp: true positive ratio
        :param fn: false negative ratio
    """
    return tp / (tp + fn)


def f1(precision, recall):
    """ Calculate F1 score
        :param precision: precision of prediction
        :param recall: recall of prediction
        :return: F1 score
    """
    return 2 * (precision * recall) / (precision + recall)


def get_precision_recall_f1(gt, pred):
    """ Get precision, recall and f1 score from prediction
        :param gt: ground truth label
        :param pred: predicted label
        :return: list [precision, recall, f1_score]
    """
    tp, tn, fp, fn = calc_pred_stats(gt, pred)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1_score = f1(prec, rec)
    return prec, rec, f1_score


def soft_dice(gt, y_hat):
    """ Soft dice coefficient
        :param gt: ground truth label (binary 0,1 mask of avalanche or not)
        :param y_hat: predicted probablity
    """
    return 2 * torch.sum(gt * y_hat) / (torch.sum(gt + y_hat))


def focal_loss(prob, target, alpha=0.5, gamma=2):
    p = prob
    q = 1 - p

    # Loss for the positive examples
    pos_loss = alpha * (q ** gamma) * -torch.log(p).clamp(min=-100)

    # Loss for the negative examples
    neg_loss = (1 - alpha) * (p ** gamma) * -torch.log(q).clamp(min=-100)

    loss = target * pos_loss + (1 - target) * neg_loss
    loss = loss.mean()
    return loss


def weighted_bce(y_hat, target, labels, weight_multiplier):
    """ Calculates the weighted BCE such that certain labels have 2x and created labels 0.5x the weight of background
     and estimated pixels.

     :param y_hat: predicted probability
     :param target: true label. 1 for avalanche 0 for background
     :param labels: like target but with certainty 1: exact, 2: estimated, 3: created
     :param weight_multiplier: Additional weight mask to be multiplied element-wise.
     """
    weight = labels.clone()
    weight.requires_grad = False
    weight[weight < 1] = 2
    weight[weight == 3] = 4
    weight = 2 / weight

    # apply mask
    weight = weight * weight_multiplier
    return F.binary_cross_entropy(y_hat, target, weight)


def per_aval_accuracy(predictions, targets, detection_thresh=(0.5, 0.7, 0.8)):
    """ Accuracy per avalanche with thresholded predictions"""
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
    """ Some useful info and soft metrics from predicted probabilities"""
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


def create_loss_weight_matrix(batch_size, patch_size, distance, min_value=0.2):
    """ Creates a matrix for weighting loss less near the patch edges. Linear interpolation is used from the patch edge
        with min_value, to 'distance' pixels inward from which the value will be 1"""
    w = min_value * torch.ones([patch_size, patch_size])
    for i in range(1, distance+1):
        value = (i + (distance-i) * min_value) / distance
        w[i:-i, i:-i] = value * torch.ones([patch_size - 2*i, patch_size - 2*i])
    w = w.expand(batch_size, 1, -1, -1).clone()
    w.requires_grad = False
    return w


def crop_to_center(tensor, border=50):
    """ Takes a torch tensor of shape BxCxHxW and crops by border pixels in the spatial dimensions"""
    return tensor[:, :, border:-border, border:-border]


if __name__ == '__main__':
    # small test
    a = torch.tensor([[0, 1, 2],
                      [0, 5, 3],
                      [0, 0, 0]])

    b = torch.tensor([[0, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]])

    tp, tn, fp, fn = calc_pred_stats(a, b)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    print(tp, tn, fp, fn)
    print('Precision: ' + str(prec))
    print('Recall: ' + str(rec))
    print('F1: ' + str(f1(prec, rec)))
