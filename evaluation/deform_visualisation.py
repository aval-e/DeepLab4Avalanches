""" This utility function can be used to visualise the deformable convolution kernel """

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt


def visualise_deformable_offsets(image, offsets, scale, pos):
    """ Visualise the offsets used in a deformable 3x3 convolution.
    Creates a plot with the image as background and the deformed convolutional kernel elements overlayed in red.
    The transformation from the standard kernel is shown with white arrows.

    :param image: background image
    :param offsets: 18 offsets
    :param scale: scale at which offsets are used relativ to input image
    :param pos: position at which offsets were used
    """

    ax = plt.subplot()

    ax.imshow(image.cpu())

    grid = np.meshgrid([-1, 0, 1], [-1, 0, 1])
    grid[0] = grid[0].flatten() * scale + pos[0]
    grid[1] = grid[1].flatten() * scale + pos[1]

    kernel = [None, None]
    kernel[0] = grid[0] + offsets[1::2].cpu().numpy() * scale
    kernel[1] = grid[1] + offsets[0::2].cpu().numpy() * scale

    ax.scatter(grid[0], grid[1], color='w')
    ax.scatter(kernel[0], kernel[1], color='r')
    ax.quiver(grid[0], grid[1], (kernel[0]-grid[0]), (kernel[1]-grid[1]), angles='xy', scale_units='xy', scale=1, color='w')

    plt.show()


if __name__ == '__main__':
    # Just some test to determine which offset elements correspond to which kernel elements and check visualisation
    a = torch.tensor([[0, 1, 2], [ 3, 4, 5], [6, 7, 8]]).view(1, 1, 3, 3).float()
    kernel = torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]).view(1, 1, 3, 3).float()

    offsets = [0, 0] + [1,0] + 14*[0]
    # offsets = torch.tensor(offsets).view(1, 18, 1, 1).float().clone()
    offsets = torch.rand(18).view(1, 18, 1, 1).float() - 0.5
    print(offsets.numpy().flatten())

    conv = torchvision.ops.DeformConv2d(1, 1, 3, bias=False)
    conv.weight = torch.nn.Parameter(kernel)

    pad = torch.nn.ZeroPad2d(1)
    a = pad(a)
    offsets = pad(offsets)

    b = conv(a, offsets)

    print(a.numpy())
    print(b)

    pos = [5, 5]
    visualise_deformable_offsets(torch.zeros([11, 11]), offsets[0, :, 1, 1], 4, pos)