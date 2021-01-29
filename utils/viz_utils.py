import matplotlib.pyplot as plt
from matplotlib import patches
from torchvision.utils import make_grid
import torch
import os
from utils.lookup import INSTLABEL_2_STR, STATUS_2_STR, STATUS_COLORS
from utils.data_utils import undo_redistribute_satellite_data


def plot_avalanches_by_certainty(image, aval_image, dem=None):
    """ Plots batch images and overlays avalanches in different colors according to their certainty. Also plot dem if
    available

    :param image: satellite image and optionally dem as torch tensor
    :param aval_images: list of 3 rasterised avalanche shapes from certain to uncertain
    :param dem: whether image includes dem or not
    """
    i = image.clone()
    min_no_channels = 4 if dem else 3
    while i.shape[3] < min_no_channels:
        i = torch.cat([i[:, :, :, 0:1], i], dim=3)
    if dem:
        dem = i[:, :, :, -1:]
        dem = (dem - dem.min()) / (dem.max() - dem.min())
    i = i[:, :, :, :3]
    i = (i - i.min()) / (i.max() - i.min())

    green = aval_image == 1
    yellow = aval_image == 2
    red = aval_image == 3

    # overlay certain avalanches in green
    i[:, :, :, 0:1] -= 0.4 * green
    i[:, :, :, 1:2] += 0.4 * green
    i[:, :, :, 2:3] -= 0.4 * green

    # overlay estimated avalanches in orange
    i[:, :, :, 0:1] += 0.4 * yellow
    i[:, :, :, 1:2] += 0.1 * yellow
    i[:, :, :, 2:3] -= 0.4 * yellow

    # overlay guessed avalanches in red
    i[:, :, :, 0:1] += 0.4 * red
    i[:, :, :, 1:2] -= 0.4 * red
    i[:, :, :, 2:3] -= 0.4 * red

    fig, axs = plt.subplots(1 if dem is None else 2, image.shape[0], squeeze=False, sharex=True, sharey=True,
                            gridspec_kw={'hspace': 0.01})
    for k in range(image.shape[0]):
        axs[0, k].imshow(i[k, :, :, :])

    if dem is not None:
        for k in range(image.shape[0]):
            axs[1, k].imshow(dem[k, :, :, :])
    plt.show()


def select_rgb_channels_from_batch(x, dem=None):
    """ Selects first 3 channels from batch to be uses as rgb values
    If less than two channels present the first channel is duplicated to make 3

    :param x: torch tensor of shape [B,C,W,H] or [C,W,H]
    :param dem: whether DEM is in x
    """
    x = x.clone()
    min_no_channels = 4 if dem else 3
    if x.ndim == 4:
        while x.shape[1] < min_no_channels:
            x = torch.cat([x[:, 0:1, :, :], x], dim=1)
        if x.shape[1] > 3:
            x = x[:, 0:3, :, :]
    elif x.ndim == 3:
        while x.shape[0] < min_no_channels:
            x = torch.cat([x[0:1, :, :], x], dim=0)
        if x.shape[0] > 3:
            x = x[0:3, :, :]
    else:
        raise Exception('Wrong number of dimensions of x')
    return x


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


def viz_predictions(x, y, y_hat, pred=None, dem=None, gt=None, fig_size=None, transpose=False):
    """ Visualise predictions during training or for qualitative evaluation

    :param x: input satellite image and may include dem
    :param y: ground truth label
    :param y_hat: nn outputs probabilities
    :param pred: thresholded predictions
    :param dem: whether dem is included in x
    :param gt: ground truth label for davos area
    :param fig_size: sequence for figure size or scalar to keep automatic aspect ratio
    :returns: matplotlib figure
    """

    with torch.no_grad():
        # if less than 3 channels, duplicate first channel for rgb image
        x_only = select_rgb_channels_from_batch(x, dem)
        x_only = undo_redistribute_satellite_data(x_only)
        x_only = (x_only - x_only.min()) / (x_only.max() - x_only.min())
        y_over = overlay_avalanches_by_certainty(x_only, y)
        y_over = y_over.clamp(0, 1)

        # convert to numpy format for plotting
        x_only = numpy_from_torch(x_only)
        y_over = numpy_from_torch(y_over)
        y_hat = numpy_from_torch(y_hat)
        if pred is not None:
            pred = numpy_from_torch(pred)

        shape = (3 if pred is None else 4, x.shape[0])
        if transpose:
            shape = (shape[1], shape[0])
        fig, axs = plt.subplots(shape[0], shape[1], sharex=True, sharey=True, squeeze=False,
                                gridspec_kw={'wspace': 0.01, 'hspace': 0.01}, facecolor='black')

        if transpose:
            axs = axs.transpose()

        j = 2
        for i in range(shape[0] if transpose else shape[1]):
            axs[0, i].imshow(x_only[i, :, :, :])
            axs[1, i].imshow(y_over[i, :, :, :])
            axs[2, i].imshow(x_only[i, :, :, :])
            if pred is not None:
                axs[2, i].imshow(pred[i, :, :, :], cmap='bwr', alpha=0.4 * pred[i, :, :, 0])
                j = 3
            axs[j, i].imshow(x_only[i, :, :, :])
            axs[j, i].imshow(y_hat[i, :, :, :], cmap=plt.cm.jet, alpha=0.5 * y_hat[i, :, :, 0])

            if gt is not None:
                axs[0, i].scatter(x.shape[1] / 2, x.shape[2] / 2, c=STATUS_COLORS[gt[i]], s=20 ** 2, marker=(5, 0),
                                  alpha=0.5)
                axs[0, i].set_title('Gt status: ' + STATUS_2_STR[gt[i]])

        # make figure aspect ratio fit content
        if not isinstance(fig_size, (list, tuple)):
            s = 1 if fig_size is None else fig_size
            fig_size = (x.shape[0], 3 if pred is None else 4)
            fig_size = (s * fig_size[0], s * fig_size[1])
            if gt is not None:
                fig_size[1] += 0.5

            if transpose:
                fig_size = [fig_size[1], fig_size[0]]

        for ax in axs.ravel():
            ax.set_axis_off()
        fig.set_size_inches(*fig_size)
        fig.subplots_adjust(0, 0, 1, 1)
        if transpose:
            fig.text(0.005, 0.01, 'SPOT6 \N{COPYRIGHT SIGN} Airbus DS2018/2019',
                     fontsize=9, color='white',
                     ha='left', va='bottom', alpha=0.7)
        else:
            fig.text(0.01, 0.98, 'SPOT6 \N{COPYRIGHT SIGN} Airbus DS2018/2019',
                     fontsize=9, color='white',
                     ha='left', va='bottom', alpha=0.7)
        return fig


def save_fig(fig, dir, name):
    """ Save a matplotlib figure"""
    fig_path = os.path.join(dir, name)
    fig.savefig(fig_path, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())


def numpy_from_torch(tensor):
    """ Utility to convert torch tensor to numpy for visualing with matplotlib"""
    if tensor.ndim == 4:
        return tensor.permute(0, 2, 3, 1).cpu().numpy()
    elif tensor.ndim == 3:
        return tensor.permute(1, 2, 0).cpu().numpy()
    return False


def viz_aval_instances(x, targets, outputs=None, dem=None, fig_size=None):
    """ Visualise outputs from instance segmentation """
    with torch.no_grad():
        fig, axs = plt.subplots(2 if outputs is None else 3, len(x), sharex=True, sharey=True, squeeze=False,
                                gridspec_kw={'wspace': 0.01, 'hspace': 0.01}, facecolor='black')

        for i in range(len(x)):
            # if less than 3 channels, duplicate first channel for rgb image
            img = select_rgb_channels_from_batch(x[i], dem)
            img = (img - img.min()) / (img.max() - img.min())

            # convert to numpy format for plotting
            img = numpy_from_torch(img)

            axs[0, i].imshow(img)
            axs[1, i].imshow(img)

            boxes = targets[i]['boxes'].cpu()
            masks = targets[i]['masks'].cpu()
            labels = targets[i]['labels'].cpu().numpy()
            label_cmap = plt.cm.get_cmap('hsv', boxes.shape[0]+1)
            for j in range(boxes.shape[0]):
                mask = masks[j, :, :]
                box = boxes[j, :]
                rect = patches.Rectangle(box[0:2], box[2] - box[0], box[3] - box[1], edgecolor=label_cmap(j), facecolor='none')
                axs[1, i].add_patch(rect)
                axs[1, i].text(box[0], box[1], INSTLABEL_2_STR[labels[j]], color=label_cmap(j))
                axs[1, i].imshow(mask, cmap=plt.cm.bwr, alpha=0.5 * mask)

            if outputs is not None:
                boxes = outputs[i]['boxes'].cpu()
                masks = numpy_from_torch(outputs[i]['masks'])
                labels = outputs[i]['labels'].cpu().numpy()
                label_cmap = plt.cm.get_cmap('hsv', boxes.shape[0] + 1)
                axs[2, i].imshow(img)
                for j in range(boxes.shape[0]):
                    mask = masks[j, :, :, :].squeeze()
                    box = boxes[j, :]
                    rect = patches.Rectangle(box[0:2], box[2] - box[0], box[3] - box[1], edgecolor=label_cmap(j), facecolor='none')
                    axs[2, i].imshow(mask, cmap=plt.cm.jet, alpha=0.5 * mask)
                    axs[2, i].add_patch(rect)
                    axs[2, i].text(box[0], box[1], INSTLABEL_2_STR[labels[j]], color=label_cmap(j))

        # make figure aspect ratio fit content
        if not isinstance(fig_size, (list, tuple)):
            s = 1 if fig_size is None else fig_size
            fig_size = (len(x), 2 if outputs is None else 3)
            fig_size = (s * fig_size[0], s * fig_size[1])

        for ax in axs.ravel():
            ax.set_axis_off()
        fig.set_size_inches(*fig_size)
        fig.subplots_adjust(0, 0, 1, 1)
        return fig
