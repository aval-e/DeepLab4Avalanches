from argparse import ArgumentTypeError
from torch.utils.data._utils.collate import default_collate
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def ba_collate_fn(batch):
    """ Collate_fn to handle batch augmentation. Expects dataset to return a list of samples if batch augmentation is used.
    If dataset returns a list of samples each of which are a list [x,y], these elements will be collated accordingly
    """
    sample_elem = batch[0]
    if isinstance(sample_elem, list):
        batch = [ba_sample for sample in batch for ba_sample in sample]
    return default_collate(batch)


def inst_collate_fn(batch):#
    # first handle batch augmentation
    sample_elem = batch[0]
    if isinstance(sample_elem, list):
        batch = [ba_sample for sample in batch for ba_sample in sample]
    # then collate into lists
    return tuple(zip(*batch))


def nanmean(x):
    """ Calculate mean ignoring nan values"""
    return x[~torch.isnan(x)].mean()
