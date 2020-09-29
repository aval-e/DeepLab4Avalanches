import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch


def viz_sample(sample):
    """ Visualise single sample of the Avalanche dataset"""
    image, aval = sample
    i = image[:, :, 0:3]
    i = (i - i.min()) / (i.max() - i.min())

    i[:, :, 0] += 0.4 * aval
    i[:, :, 1] -= 0.4 * aval
    i[:, :, 2] -= 0.4 * aval
    plt.imshow(i)
    plt.show()

    # also plot DEM if available
    if (image.shape[2] == 5):
        dem = image[:, :, 4]
        plt.imshow(dem)
        plt.show()


def overlay_and_plot_avalanches_by_certainty(image, aval_image):
    """ Plots image and overlays avalanches in different colors according to their certainty

    :param image: background satellite image as numpy array
    :param aval_images: list of 3 rasterised avalanche shapes from certain to uncertain
    """
    i = image[:, :, 0:3].clone()
    i = (i - i.min()) / (i.max() - i.min())

    green = aval_image == 1
    yellow = aval_image == 2
    red = aval_image == 3

    # overlay certain avalanches in green
    i[:, :, 0] -= 0.4 * green
    i[:, :, 1] += 0.4 * green
    i[:, :, 2] -= 0.4 * green

    # overlay estimated avalanches in orange
    i[:, :, 0] += 0.4 * yellow
    i[:, :, 1] += 0.1 * yellow
    i[:, :, 2] -= 0.4 * yellow

    # overlay guessed avalanches in red
    i[:, :, 0] += 0.4 * red
    i[:, :, 1] -= 0.4 * red
    i[:, :, 2] -= 0.4 * red

    plt.imshow(i)
    plt.show()

    # also plot DEM if available
    if (image.shape[2] == 5):
        dem = image[:, :, 4]
        plt.imshow(dem)
        plt.show()


def viz_training(x, y, y_hat, pred=None):
    """
    Show input, ground truth and prediction next to each other.

    All arguments are torch tensors with [B,C,H,W]
    :param x: satellite image
    :param y: ground truth
    :param y_hat: probability output
    :param pred: prediction - y_hat rounded to zero or one
    :return: image grid of comparisons for all samples in batch
    """
    with torch.no_grad():
        # if less than 3 channels, duplicate first channel for rgb image
        if x.shape[1] >= 3:
            x_only = x[:, 0:3, :, :]
        elif x.shape[1] == 2:
            x_only = torch.cat([x, x[:, 0:1, :, :]], dim=1)
        else:
            x_only = torch.cat([x, x[:, 0:1, :, :], x[:, 0:1, :, :]], dim=1)

        x_only = (x_only - x_only.min()) / (x_only.max() - x_only.min())
        y_over = overlay_avalanches_by_certainty(x_only, y)
        y_hat_over = overlay_avalanches(x_only, y_hat)
        if pred is not None:
            pred_over = overlay_avalanches(x_only, pred)
            image_list = [x_only, y_over, pred_over, y_hat_over]
        else:
            image_list = [x_only, y_over, y_hat_over]

        image_array = torch.cat(image_list, dim=0)
        image = make_grid(image_array, nrow=x.shape[0])
    return image.clamp(0, 1)


def overlay_avalanches(image, aval_image):
    """
    Overlays avalanche image on satellite image and returns image.
    Expects torch tensors as inputs. If Image has 4 channels only uses first 3.
    If input is a batch will do the same for all samples

    :param image: Satellite image. If 4 channels only uses first 3
    :param aval_image: image mask of where avalanche is.
    :return: image of with avalanches overlayed in red
    """

    if torch.is_tensor(image):
        with torch.no_grad():
            if image.dim() == 3:
                i = image[0:3, :, :].clone()
                i[0:1, :, :] += 0.5 * aval_image
                i[1:2, :, :] -= 0.5 * aval_image
                i[2:3, :, :] -= 0.5 * aval_image
            else:
                i = image[:, 0:3, :, :].clone()
                i[:, 0:1, :, :] += 0.5 * aval_image
                i[:, 1:2, :, :] -= 0.5 * aval_image
                i[:, 2:3, :, :] -= 0.5 * aval_image
    else:
        if image.ndim == 3:
            i = image[:, :, 0:3].clone()
            i[:, :, 0] += 0.5 * aval_image
            i[:, :, 1] -= 0.5 * aval_image
            i[:, :, 2] -= 0.5 * aval_image
        else:
            i = image[:, :, :, 0:3].clone()
            i[:, :, :, 0] += 0.5 * aval_image
            i[:, :, :, 1] -= 0.5 * aval_image
            i[:, :, :, 2] -= 0.5 * aval_image
    return i


def overlay_avalanches_by_certainty(image, aval_image):
    """ Overlay avalanches onto image for batch of torch tensors
        :param image: optical satellite image
        :param aval_image: rasterised avalanches consistent of 1 layer with value corresponding to avalanche certainty
        :returns: image
    """
    with torch.no_grad():
        green = aval_image == 1
        yellow = aval_image == 2
        red = aval_image == 3

        i = image[:, 0:3, :, :].clone()
        i[:, 0:1, :, :] -= 0.4 * green
        i[:, 1:2, :, :] += 0.4 * green
        i[:, 2:3, :, :] -= 0.4 * green

        i[:, 0:1, :, :] += 0.4 * yellow
        i[:, 1:2, :, :] += 0.1 * yellow
        i[:, 2:3, :, :] -= 0.4 * yellow

        i[:, 0:1, :, :] += 0.4 * red
        i[:, 1:2, :, :] -= 0.4 * red
        i[:, 2:3, :, :] -= 0.4 * red

        return i