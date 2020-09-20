import torch


def calc_pred_stats(gt, pred):
    """ Calculate the true positive, true negative, false positive and false negative ratios of a mask of 0s ans 1s.
        The statistics of interest are with regard to the 1 label

        :param gt: ground truth values
        :param pred: predicated values
        :return: list of [tp, tn, fp, fn]
    """
    t = gt == pred
    f = ~t
    n = float(gt.numel())
    tp = torch.sum(pred[t]) / n
    tn = torch.sum(t) / n - tp
    fp = torch.sum(pred[f]) / n
    fn = torch.sum(f) / n - fp
    return tp, tn, fp, fn


def precision(tp, fp):
    """ Calculate the precision
        :param tp: true positive ratio
        :param fp: false positive ratio
    """
    if tp + fp == 0:
        return torch.tensor(1.0).type_as(tp)
    return tp / (tp + fp)


def recall(tp, fn):
    """ Calculate the recall
        :param tp: true positive ratio
        :param fn: false negative ratio
    """
    if tp + fn == 0:
        return torch.tensor(1.0).type_as(tp)
    return tp / (tp + fn)


def f1(precision, recall):
    """ Calculate F1 score
        :param precision: precision of prediction
        :param recall: recall of prediction
        :return: F1 score
    """
    if precision + recall == 0:
        return torch.tensor(0.0).type_as(recall)
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


if __name__ == '__main__':
    # small test
    a = torch.tensor([[0, 1, 1],
                      [0, 1, 1],
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

