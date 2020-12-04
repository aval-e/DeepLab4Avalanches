import torch.nn as nn
from modeling.reusable_blocks import Bottleneck, DeformableBlock, BasicBlock, conv1x1, conv3x3


class CustomResNet(nn.Module):

    def __init__(self, block=Bottleneck, layers=(3, 4, 6, 3), zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=(False, False, True),
                 norm_layer=None, out_channels=(3, 64, 256, 512, 1024, 2048), deformable=False):
        super(CustomResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.out_channels = out_channels
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, deformable=False):
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        layers = []
        layers.append(block(self.inplanes, planes, stride, self.groups,
                            self.base_width, previous_dilation, norm_layer, deformable))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        stages = self.get_stages()

        features = []
        for stage in stages:
            x = stage(x)
            features.append(x)

        return features

    def forward(self, x):
        return self._forward_impl(x)


def resnet_standard():
    return CustomResNet()


def resnet_deformable():
    return CustomResNet(block=DeformableBlock)


def resnet_leaky():
    avanet = resnet_standard()
    convert_layer(avanet, nn.ReLU, nn.LeakyReLU, inplace=True)
    return avanet


def resnet_small():
    avanet = CustomResNet(layers=(3, 2, 2, 2))
    return avanet


def convert_layer(model, old_layer, new_layer, **kwargs):
    for child_name, child in model.named_children():
        if isinstance(child, old_layer):
            setattr(model, child_name, new_layer(**kwargs))
        else:
            convert_layer(child, old_layer, new_layer, **kwargs)
