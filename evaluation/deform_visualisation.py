import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt


def visualise_deformable_offsets(image, offsets, scale, pos):
    """ Visualise the offsets used in a deformable 3x3 convolution

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

    ax.scatter(grid[0], grid[1])
    ax.scatter(kernel[0], kernel[1], color='r')
    ax.quiver(grid[0], grid[1], (kernel[0]-grid[0]), (kernel[1]-grid[1]), angles='xy', scale_units='xy', scale=1, color='g')

    plt.show()

if __name__ == '__main__':
    a = torch.tensor([[0, 1, 2], [ 3, 4, 5], [6, 7, 8]]).view(1, 1, 3, 3).float()
    kernel = torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]).view(1, 1, 3, 3).float()

    offsets = [0, 0] + [-0.5, 0] + 14*[0]
    # offsets = torch.tensor(offsets).view(1, 18, 1, 1).float().clone()
    offsets = torch.rand(18).view(1, 18, 1, 1).float()
    print(offsets.numpy().flatten())

    conv = torchvision.ops.DeformConv2d(1, 1, 3, bias=False)
    conv.weight = torch.nn.Parameter(kernel)

    b = conv(a, offsets)

    print(a.numpy())
    print(b.item())

    pos = [5, 5]
    visualise_deformable_offsets(torch.zeros([11, 11]), offsets[0, :, 0, 0], 1, pos)