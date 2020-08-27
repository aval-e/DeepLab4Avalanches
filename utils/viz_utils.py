import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch


def viz_sample(sample):
    """ Visualise single sample of the Avalanche dataset"""
    image, aval = sample
    i = image[:, :, 0:3]
    i[:, :, 0] += 0.4 * aval
    i[:, :, 1] -= 0.4 * aval
    i[:, :, 2] -= 0.4 * aval

    plt.imshow(i)
    plt.show()


def overlay_and_plot_avalanches_by_certainty(image, aval_images):
    """ Plots image and overlays avalanches in different colors according to their certainty

    :param image: background satellite image as numpy array
    :param aval_images: list of 3 rasterised avalanche shapes from certain to uncertain
    """
    i = image[:, :, 0:3]

    # overlay certain avalanches in green
    i[:, :, 0] -= 0.4 * aval_images[0]
    i[:, :, 1] += 0.4 * aval_images[0]
    i[:, :, 2] -= 0.4 * aval_images[0]

    # overlay estimated avalanches in orange
    i[:, :, 0] += 0.4 * aval_images[1]
    i[:, :, 1] += 0.1 * aval_images[1]
    i[:, :, 2] -= 0.4 * aval_images[1]

    # overlay guessed avalanches in red
    i[:, :, 0] += 0.4 * aval_images[2]
    i[:, :, 1] -= 0.4 * aval_images[2]
    i[:, :, 2] -= 0.4 * aval_images[2]

    plt.imshow(image)
    plt.show()


def viz_training(x, y, y_hat):
    """
    Show input, ground truth and prediction next to each other.

    All arguments are torch tensors with [B,C,H,W]
    :param x: satellite image
    :param y: ground truth
    :param y_hat: prediction
    :return: image grid of comparisons for all samples in batch
    """
    with torch.no_grad():
        x_only = x[:,0:3,:,:]
        y_over = overlay_avalanches(x_only, y)
        y_hat_over = overlay_avalanches(x_only, y_hat)
        image_array = torch.cat([x_only, y_over, y_hat_over], dim=0)
        image = make_grid(image_array, nrow=x.shape[0])
    return image.clamp(0,1)


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
                i = image[0:3,:,:].clone()
                i[0:1, :, :] += 0.4 * aval_image
                i[1:2, :, :] -= 0.4 * aval_image
                i[2:3, :, :] -= 0.4 * aval_image
            else:
                i = image[:, 0:3, :, :].clone()
                i[:, 0:1, :, :] += 0.4 * aval_image
                i[:, 1:2, :, :] -= 0.4 * aval_image
                i[:, 2:3, :, :] -= 0.4 * aval_image
    else:
        if image.ndim == 3:
            i = image[:, :, 0:3].clone()
            i[:, :, 0] += 0.4 * aval_image
            i[:, :, 1] -= 0.4 * aval_image
            i[:, :, 2] -= 0.4 * aval_image
        else:
            i = image[:, :, :, 0:3].clone()
            i[:, :, :, 0] += 0.4 * aval_image
            i[:, :, :, 1] -= 0.4 * aval_image
            i[:, :, :, 2] -= 0.4 * aval_image
    return i