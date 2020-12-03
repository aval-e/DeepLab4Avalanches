import torch
import torch.nn as nn
from torch.nn.functional import grid_sample
from modeling.reusable_blocks import Bottleneck, BasicBlock, conv1x1, conv3x3
from kornia.filters.sobel import SpatialGradient


class GridSampleNet(nn.Module):

    def __init__(self, groups=1, width_per_group=64,
                 norm_layer=None, iterations=50):
        super(GridSampleNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        self.iterations = iterations
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = self._make_layer(Bottleneck, self.inplanes, 64, 3, stride=1)
        self.layer3 = self._make_layer(Bottleneck, 64*4, 128, 6, stride=2)

        self.spatial_grad = SpatialGradient()
        self.grad_scale = nn.Parameter(torch.rand(1))
        self.avgpool = nn.AvgPool2d(4)
        self.feature_squeeze = conv1x1(128*Bottleneck.expansion, 64)

        merge_outplanes = 128 * Bottleneck.expansion
        self.bn3 = norm_layer(merge_outplanes + 2*64)
        self.tanh = nn.Tanh()
        self.merge = conv1x1(merge_outplanes + 2*64, merge_outplanes)
        self.postprocess = self._make_layer(Bottleneck, merge_outplanes, 128, 4, stride=2)
        self.out_channels = [merge_outplanes, 0, 0, merge_outplanes]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer

        layers = []
        layers.append(block(inplanes, planes, stride, self.groups,
                            self.base_width, 1, norm_layer))
        for _ in range(1, blocks):
            layers.append(block(planes*block.expansion, planes, groups=self.groups,
                                base_width=self.base_width, dilation=1,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.conv2, self.bn2, self.relu, self.maxpool),
            self.layer2,
            self.layer3,
        ]

    def forward(self, x):
        features = x[:, :-1, :, :]
        dem = x[:, [-1], :, :]

        # calculate dem gradients and magnitude
        dem_grads = self.spatial_grad(dem).squeeze(dim=1)
        dem = torch.sqrt(torch.square(dem_grads[:, [0], :, :]) + torch.square(dem_grads[:, [1], :, :]))
        features = torch.cat([features, dem], dim=1)

        stages = self.get_stages()

        for stage in stages:
            features = stage(features)

        dem_grads = self.avgpool(dem_grads)
        dem_grads *= self.grad_scale

        dem_grads *= dem_grads.shape[2] / 64  # rescale to work with any patch size since grid is relativ to patch size -
        grid1 = dem_grads.permute(0, 2, 3, 1)
        grid2 = -grid1

        # propagate features along dem gradient field
        out = self.feature_squeeze(features)
        out_dir1 = out
        out_dir2 = out
        sum_dir1 = torch.zeros_like(out)
        sum_dir2 = torch.zeros_like(out)
        for i in range(self.iterations):
            out_dir1 = grid_sample(out_dir1, grid1)
            out_dir2 = grid_sample(out_dir2, grid2)
            sum_dir1 += out_dir1
            sum_dir2 += out_dir2
        out = torch.cat([features, sum_dir1, sum_dir2], dim=1)
        out = self.bn3(out)
        out = self.tanh(out)
        out = self.merge(out)
        out = self.postprocess(out)

        return [features, None, None, out]
