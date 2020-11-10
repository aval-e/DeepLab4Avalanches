from scipy import ndimage
from torchvision.transforms import functional as F
import torch
import random
import numbers


class RandomScaling:
    """ Random scaling of a torch tensor. Simulates something like a change in contrast
    :param max_scale: scale chosen from a triangular distribution between 1-max_scale and 1+max_scale"""

    def __init__(self, max_scale):
        self.max_scale = max_scale

    @staticmethod
    def get_params(max_scale):
        return random.triangular(1 - max_scale, 1 + max_scale, 1)

    def __call__(self, img):
        return img * self.get_params(self.max_scale)


class RandomShift:
    """ Random shifting of a torch tensor. Simulates something like a change in brightness
    :param max_shift: shift chosen from a triangular distribution between -max_shift and max_shift"""

    def __init__(self, max_shift):
        self.max_shift = max_shift

    @staticmethod
    def get_params(max_shift):
        return random.triangular(-max_shift, max_shift, 0)

    def __call__(self, img):
        return img - self.get_params(self.max_shift)


class RandomHorizontalFlip:
    """Random horizontal flip
    :param p (float): probability of horizontal flip. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def get_param(self):
        """ Get a random value within the objects bounds """
        p = random.random()
        return p < self.p

    def __call__(self, img, flip=None):
        if flip is None:
            flip = self.get_param()
        if flip:
            return F.hflip(img)
        return img


class RandomRotation:
    """ Random rotation of numpy ndarray

        :param degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    def get_param(self):
        """ Get a random value within the objects bounds """
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return angle

    def __call__(self, img, angle=None):
        """ returns a randomly rotated version of the input"""
        if not angle:
            angle = self.get_param()

        return ndimage.rotate(img, angle, reshape=False, order=1)


def center_crop_batch(batch, crop_size=11):
    """ crop each element in batch tensor at the center
    :param batch: tensor batch [BxCxWxH]
    :param crop_size: size in pixels of crop taken at center
    """
    assert(batch.ndim == 4)

    cropped = []
    for img in batch:
        cropped.append(center_crop(img, crop_size))
    return torch.stack(cropped, dim=0)


def center_crop(img, crop_size=11):
    """ crop tensor at the center. If patch cannot be centered exactly top left option is chosen.
    :param img: tensor image [CxWxH]
    :param crop_size: size in pixels of crop taken at center
    """
    crop_size = torch.tensor([crop_size, crop_size])
    s = torch.tensor(img.shape[1:3])
    center = (s - 1) * 0.5
    padding = (crop_size - 1) * 0.5
    low_bound = (center - padding).floor().int().tolist()
    upper_bound = (center + padding).floor().int().tolist()
    return img[:, low_bound[0]:upper_bound[0], low_bound[1]:upper_bound[1]]

