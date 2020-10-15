from scipy import ndimage
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

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])
        return angle

    def __call__(self, img):
        """ returns a randomly rotated version of the input"""
        angle = self.get_params(self.degrees)

        return ndimage.rotate(img, angle, reshape=False, order=1)


def center_crop_batch(batch, fac=0.8):
    """ crop each element in batch tensor at the center
    :param fac: size with respect to original 1: return original, 0: return nothing
    """
    assert(batch.ndim == 4)

    cropped = []
    for img in batch:
        cropped.append(center_crop(img, fac))
    return torch.stack(cropped, dim=0)


def center_crop(img, fac=0.8):
    """ crop tensor at the center
    :param fac: size with respect to original 1: return original, 0: return nothing
    """
    s = torch.tensor(img.shape[1:3])
    w, h = (s * 0.5 * (1 - fac)).int()
    return img[:, w:-w, h:-h]

