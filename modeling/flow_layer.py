import torch
from torch import nn
from modeling.reusable_blocks import conv1x1, conv3x3, SeparableConv2d, Bottleneck
from kornia.filters.sobel import SpatialGradient
from kornia.utils.image import image_to_tensor, tensor_to_image
from kornia.enhance.normalize import normalize_min_max
from kornia.augmentation import RandomCrop
import matplotlib.pyplot as plt
import time


class FlowLayer(nn.Module):
    """ Layer which implements flow propagation along a gradient field in both directions.

    Propagates features up and down a gradient field. Can cover a larger receptive field than standard convolution operation.
    To get a smooth output and reduce computation, features are propagated at a reduced resolution such that they are
    moved one pixel at a time. The output is upsampled bilinearly.

    :param inplanes: no. of input channels
    :param outplanes: no. of outputs channels
    :param iterations: how many flow iterations to compute
    :param pixels_per_iter: how many pixels to propagate features by each iteration
    """

    def __init__(self, inplanes, outplanes, iterations=10, pixels_per_iter=4):
        super().__init__()
        self.iters = iterations
        self.avgdown = nn.AvgPool2d(pixels_per_iter)
        self.register_buffer('theta', torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(dim=0).float())

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.flow1 = FlowProcess(inplanes, outplanes, iterations, pixels_per_iter)
        self.flow2 = FlowProcess(inplanes, outplanes, iterations, pixels_per_iter)

        self.merge = SeparableConv2d(2 * outplanes, outplanes, 3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=pixels_per_iter)

    def forward(self, x, grads):
        # get grads in absolute terms such that results remain independent of input size
        grads = self.avgdown(grads)  # keep gradient direction when downsampling
        grid_shape = grads.shape
        grads = grads / grid_shape[2]
        grads = grads.permute(0, 2, 3, 1).contiguous()

        # compute absolute sample points from relativ offsets (grads)
        grid = nn.functional.affine_grid(self.theta.expand(x.shape[0], 2, 3), grid_shape, align_corners=True)
        grid1 = grid + grads
        grid2 = grid - grads

        # Iterate flows in both up and down hill
        x = self.bn1(x)
        x1 = self.flow1(x, grid1)
        x2 = self.flow2(x, grid2)

        # combine results
        x = torch.cat([x1, x2], dim=1)
        x = x / self.iters  # ensure the same statistics independent of no. iterations
        x = self.merge(x)
        x = self.up(x)
        return x


class FlowProcess(nn.Module):
    """ Flow process which is required once in each direction by Flowlayer"""

    def __init__(self, inplanes, outplanes, iterations=10, pixels_per_iter=4):
        super().__init__()
        self.iters = iterations
        self.maxdown = nn.MaxPool2d(pixels_per_iter)
        self.conv = conv1x1(inplanes, outplanes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.att = SeparableConv2d(inplanes, outplanes, 3, padding=1)
        self.add = conv1x1(2*outplanes, outplanes)

    def forward(self, x, grid):
        # Compute attention map based on input features
        attention = self.sigmoid(self.att(x))
        x = self.conv(x)

        # keep avalanche features even if they are small with max pooling
        attention = self.maxdown(attention)
        x = self.maxdown(x)

        # Relu instead of sigmoid to avoid saturating gradients
        x = self.relu(x)
        aggregate = x
        for _ in range(self.iters):
            x = nn.functional.grid_sample(x, grid, align_corners=True)
            x = x * attention  # kill inputs when they move past specific locations
            aggregate = self.add(torch.cat([aggregate, x], dim=1))
        return aggregate


class FlowAttention(nn.Module):
    """ Attention Layer for where to propagate information along gradient"""

    def __init__(self, inplanes, replace_stride_with_dilation=False):
        super().__init__()
        self.replace_stride_with_dilation = replace_stride_with_dilation
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.block1 = Bottleneck(inplanes[-1] + inplanes[-2], inplanes[-2])
        self.block2 = Bottleneck(inplanes[-2] + inplanes[-3], inplanes[-3])
        self.conv1x1 = conv1x1(inplanes[-3], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.upsample(x[-1]) if not self.replace_stride_with_dilation else x[-1]
        features = torch.cat([features, x[-2]], dim=1)
        features = self.block1(features)
        features = self.upsample(features)
        features = torch.cat([features, x[-3]], dim=1)
        features = self.block2(features)
        features = self.conv1x1(features)
        features = self.sigmoid(features)
        return torch.cat(2 * [features], dim=1)


if __name__ == '__main__':
    plt.interactive(True)
    import matplotlib
    matplotlib.use('TkAgg')

    spatial_grad = SpatialGradient()
    flowlayer = FlowLayer(3, 3, pixels_per_iter=8)
    tile = 256
    crop = RandomCrop([tile, tile])

    img = plt.imread('/home/patrick/Pictures/SpaceBok-2-edited.jpg')
    img = img.copy()

    img = image_to_tensor(img, keepdim=False).float()
    img = crop(img)
    img = normalize_min_max(img)

    plt.imshow(tensor_to_image(img))
    plt.show()

    with torch.no_grad():
        # dem = img[:, [0], :, :]
        # dem = torch.linspace(-1, 1, tile)
        # dem = torch.stack(tile*[dem], dim=0).unsqueeze(dim=0).unsqueeze(dim=0)

        grads = torch.stack(torch.meshgrid([torch.logspace(-1, 1, tile), torch.linspace(1, -1, tile)]), dim=0).unsqueeze(dim=0)
        # dem = grads.mean(dim=1, keepdim=True)
        # plt.imshow(tensor_to_image(dem))
        # plt.show()
        # grads = spatial_grad(dem).squeeze(dim=1)
        grads = grads + 1e-5  # avoid dividing by zero
        grads = grads / grads.norm(p=None, dim=1, keepdim=True)

        grad_viz = torch.cat([grads, torch.ones_like(grads[:, [0], :, :])], dim=1)
        plt.imshow(tensor_to_image(grad_viz))
        plt.show()

        start = time.time()
        x = flowlayer(img, grads)
        print(time.time() - start)
