import torch
from torch import nn
from modeling.reusable_blocks import conv1x1, conv3x3, SeparableConv2d, Bottleneck
from kornia.filters.sobel import SpatialGradient
from kornia.utils.image import image_to_tensor, tensor_to_image
from kornia.enhance.normalize import normalize_min_max
from kornia.augmentation import RandomCrop
import matplotlib.pyplot as plt


class FlowLayer(nn.Module):
    """ Layer which implements flow propagation along a gradient field in both directions"""

    def __init__(self, inplanes, outplanes, pixels_per_iter=4):
        super().__init__()
        self.pixels_per_iter = pixels_per_iter
        self.register_buffer('theta', torch.tensor([[1, 0, 0], [0, 1, 0]]).unsqueeze(dim=0).float())
        self.conv1 = conv1x1(inplanes, outplanes)
        self.conv2 = conv1x1(inplanes, outplanes)
        self.sigmoid = nn.Sigmoid()
        self.merge = SeparableConv2d(2 * outplanes, outplanes, 3, padding=1)
        self.postprocess = SeparableConv2d(outplanes, outplanes, 3, padding=1)

    def forward(self, x, grads):
        # Only propagate features a maximum of 3/4 of the image size since more would be overkill
        iters = (3 * x.shape[2]) // (4 * self.pixels_per_iter)
        grads = 1.5 * grads / iters
        grads = grads.permute(0, 2, 3, 1).contiguous()

        # compute absolute sample points from relativ offsets (grads)
        grid = nn.functional.affine_grid(self.theta.expand(x.shape[0], 2, 3), x.size(), align_corners=True)
        grid1 = grid + grads
        grid2 = grid - grads

        m1 = self.conv1(x)
        m2 = self.conv2(x)
        m1 = self.sigmoid(m1)
        m2 = self.sigmoid(m2)
        m1_sum = m1
        m2_sum = m2
        for _ in range(iters):
            m1 = nn.functional.grid_sample(m1, grid1, align_corners=True)
            m2 = nn.functional.grid_sample(m2, grid2, align_corners=True)
            m1_sum = m1_sum + m1
            m2_sum = m2_sum + m2
        m1 = self.sigmoid(m1_sum)
        m2 = self.sigmoid(m2_sum)
        x = torch.cat([m1, m2], dim=1)
        x = self.merge(x)
        return self.postprocess(x)


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

        x = flowlayer(img, grads)
