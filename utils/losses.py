import torch


def calc_pred_stats(y, pred):
    t = y == pred
    f = ~t
    n = float(y.numel())
    tp = torch.sum(pred[t]) / n
    tn = torch.sum(t) / n - tp
    fp = torch.sum(pred[f]) / n
    fn = torch.sum(f) / n - fp
    return tp, tn, fp, fn


def precision(tp, fp):
    return tp / (tp + fp)


def recall(tp, fn):
    return tp / (tp + fn)


def f1(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def get_precision_recall_f1(y, pred):
    tp, tn, fp, fn = calc_pred_stats(y, pred)
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

