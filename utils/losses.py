import torch


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


def focal_loss(input, target, alpha, gamma):
    EPSILON = 1e-10
    p = input
    q = 1 - p

    # avoid log of 0
    p.clamp(min=EPSILON, max=1)
    q.clamp(min=EPSILON, max=1)

    # Loss for the positive examples
    pos_loss = -alpha * (q ** gamma) * torch.log(p)

    # Loss for the negative examples
    neg_loss = -(1 - alpha) * (p ** gamma) * torch.log(q)

    loss = target * pos_loss + (1 - target) * neg_loss
    loss = loss.mean()

    return loss


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
