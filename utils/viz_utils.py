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


def plot_avalanches_by_certainty(image, aval_image):
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


def select_rgb_channels_from_batch(x, dem=None):
    """ Selects first 3 channels from batch to be uses as rgb values
    If less than two channels present the first channel is duplicated to make 3

    :param x: torch tensor of shape [B,C,W,H]
    :param dem: whether DEM is in x
    """
    x = x.clone()
    min_no_channels = 3
    if dem:
        min_no_channels += 2
    while x.shape[1] < min_no_channels:
        x = torch.cat([x[:,0:1,:,:], x], dim=1)
    if x.shape[1] > 3:
        x = x[:,0:3,:,:]
    return x


def viz_training(x, y, y_hat, pred=None, dem=None):
    """
    Show input, ground truth and prediction next to each other.

    All arguments are torch tensors with [B,C,H,W]
    :param x: satellite image
    :param y: ground truth
    :param y_hat: probability output
    :param pred: prediction - y_hat rounded to zero or one
    :param dem: whether DEM is in x
    :return: image grid of comparisons for all samples in batch
    """
    with torch.no_grad():
        # if less than 3 channels, duplicate first channel for rgb image
        x_only = select_rgb_channels_from_batch(x, dem)

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

        return i.clamp(0, 1)


def plot_prediction(image, y, y_hat):
    with torch.no_grad():
        image = (image - image.min()) / (image.max() - image.min())
        image = select_rgb_channels_from_batch(image)
        y_over = overlay_avalanches_by_certainty(image, y)

        # convert to numpy format for plotting
        image = image.squeeze().permute(1, 2, 0).numpy()
        y_over = y_over.squeeze().permute(1, 2, 0).numpy()
        y_hat = y_hat.squeeze().numpy()

        alpha_map = 0.5 * y_hat

        fig, axs = plt.subplots(1, 3, sharey=True, gridspec_kw={'wspace': 0.01})
        axs[0].imshow(image)
        axs[1].imshow(y_over)
        axs[2].imshow(image)
        axs[2].imshow(y_hat, cmap=plt.cm.jet, alpha=alpha_map)
        for ax in axs:
            ax.axis('off')
        fig.set_size_inches(12,4)
        fig.subplots_adjust(0,0,1,1)
        fig.show()
        return fig
