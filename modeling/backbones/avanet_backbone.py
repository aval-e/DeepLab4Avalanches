import torch
import torch.nn as nn
from torch.utils import model_zoo
from segmentation_models_pytorch.encoders.resnet import resnet_encoders
from modeling.reusable_blocks import Bottleneck, DeformableBlock, SeBlock, BasicBlock
from torchvision.ops.deform_conv import DeformConv2d
from torchvision.models.resnet import ResNet, conv1x1
from segmentation_models_pytorch.encoders import _utils as utils


class AvanetBackbone(nn.Module):

    def __init__(self, groups=1, width_per_group=64, norm_layer=None, replace_stride_with_dilation=False,
                 no_blocks=(2, 3, 2, 2), deformable=True):
        super(AvanetBackbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.groups = groups
        self.base_width = width_per_group

        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 62, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(62, 62, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(62, 62, kernel_size=3, padding=1, bias=False),
        )
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ch = (3, 64, 64, 128, 256, 512)
        self.out_channels = ch

        self.layer1 = self._make_layer(ch[1], ch[2], no_blocks[0], stride=2, deformable=deformable)
        self.layer2 = self._make_layer(ch[2], ch[3], no_blocks[1], stride=2, deformable=deformable)
        self.layer3 = self._make_layer(ch[3], ch[4], no_blocks[2], stride=2, deformable=deformable)
        if replace_stride_with_dilation:
            self.layer4 = self._make_layer(ch[4], ch[5], no_blocks[3], dilation=2, deformable=deformable)
        else:
            self.layer4 = self._make_layer(ch[4], ch[5], no_blocks[3], stride=2, deformable=deformable)

        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, inplanes, planes, blocks, stride=1, dilation=1, deformable=False):
        norm_layer = self._norm_layer
        block = Bottleneck if not deformable else DeformableBlock

        layers = []
        layers.append(block(inplanes, planes, stride, 1,
                            self.base_width, dilation, norm_layer))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=1,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, grads):
        features = [nn.Identity()]

        x = self.layer0(x)
        x = torch.cat([x, grads], dim=1)
        x = self.bn1(x)
        x = self.relu(x)
        features.append(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return features


class AdaptedResnet(ResNet):
    """ Version of the resnet that uses deformable convolutions in all its layers.

     Offsets for the deformable convolutions are calculated by a seperate network.
     :param grad_feats: general gradient features as calculated by another network
     :param in_channels: number of input channels including the DEM if used
     :param dem: whether one of the input channels represents the DEM. Parameters will be initialised differently for
        the DEM channel.
     :param backbone: Resnet backbone to be adapted. Works for resnet18, resnet34 and resnet50
     :param depth: Depth of the network
     :param replace_stride_with_dilation: whether to replace the stride in the last layer with dilation. Results in a
        higher resolution output but also required more computation.
     :param pretrained: whether to use pretrained weights
     """
    def __init__(self, grad_feats, in_channels=3, dem=True, backbone='resnet18', depth=5,
                 replace_stride_with_dilation=True, pretrained=True):
        super().__init__(block=resnet_encoders[backbone]["params"]['block'],
                         layers=resnet_encoders[backbone]["params"]['layers'])
        if pretrained:
            settings = resnet_encoders[backbone]["pretrained_settings"]['imagenet']
            self.load_state_dict(model_zoo.load_url(settings["url"]))

        self._depth = depth
        self.out_channels = resnet_encoders[backbone]["params"]['out_channels']
        self._in_channels = in_channels

        self._setup_first_conv(in_channels, dem)

        del self.fc
        del self.avgpool

        self.stages = nn.ModuleList([
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ])

        if replace_stride_with_dilation:
            self._make_dilated(
                stage_list=[5],
                dilation_list=[2],
            )

        # make all blocks in layers deformable
        layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i in range(len(layers)):
            for j in range(len(layers[i])):
                layers[i][j] = DeformableBasicBlock(layers[i][j])

        self.offsetnet = OffsetNet(grad_feats, replace_stride_with_dilation)

    def forward(self, x, grad_feats):
        offsets = self.offsetnet(grad_feats)

        features = []
        for i in range(2):
            x = self.stages[i](x)
            features.append(x)

        x = self.maxpool(x)
        x, _ = self.layer1([x, offsets[0]])
        features.append(x)

        x, _ = self.layer2([x, offsets[1]])
        features.append(x)

        x, _ = self.layer3([x, offsets[2]])
        features.append(x)

        x, _ = self.layer4([x, offsets[3]])
        features.append(x)

        return features

    def _make_dilated(self, stage_list, dilation_list):
        stages = self.stages
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            utils.replace_strides_with_dilation(
                module=stages[stage_indx],
                dilation_rate=dilation_rate,
            )

    def _setup_first_conv(self, in_channels, dem=True):
        """ Setup first convolution for network to work with any number of input channels. Use pretrained weights of
        those channels available duplicating them if more channels are used.

        Make the weights corresponding the DEM channel random rather than pretrained, due to its different distribution
        statistics compared to rgb images.

        :param in_channels: total number of input channels
        :param dem: whether the last input channel corresponds to the DEM
        """
        new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            if in_channels <= 3:
                new_conv1.weight[:, :, :, :] = self.conv1.weight[:, 0:in_channels, :, :]
            else:
                new_conv1.weight[:, 0:3, :, :] = self.conv1.weight
                for i in range(3, in_channels):
                    new_conv1.weight[:, i, :, :] = new_conv1.weight[:, 0, :, :]

            # make dem channel use a random distribution
            if dem:
                dem_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                nn.init.kaiming_normal_(dem_conv.weight, mode='fan_out', nonlinearity='relu')
                new_conv1.weight[:, [-1], :, :] = dem_conv.weight

        self.conv1 = new_conv1


class DeformableBasicBlock(nn.Module):
    def __init__(self, basic_block):
        super(DeformableBasicBlock, self).__init__()
        conv1 = basic_block.conv1
        deformconv = DeformConv2d(conv1.in_channels, conv1.out_channels, conv1.kernel_size, conv1.stride,
                                  conv1.dilation, conv1.dilation, conv1.groups, conv1.bias)
        deformconv.weight.data.copy_(conv1.weight.data)

        self.conv1 = deformconv
        self.bn1 = basic_block.bn1
        self.relu = basic_block.relu
        self.conv2 = basic_block.conv2
        self.bn2 = basic_block.bn2
        self.downsample = basic_block.downsample
        self.stride = basic_block.stride

    def forward(self, x):
        x, offsets = x
        identity = x

        out = self.conv1(x, offsets)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return (out, offsets)


class SeDeformableBasicBlock(DeformableBasicBlock):
    def __init__(self, basic_block):
        super().__init__(basic_block=basic_block)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.conv2.out_channels, self.conv2.out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, offsets = x
        identity = x

        out = self.conv1(x, offsets)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        b, c, _, _ = out.shape
        se = self.pool(out).view(b, c)
        se = self.fc(se)
        se = self.sigmoid(se).view(b, c, 1, 1)
        out = out * se.expand_as(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return (out, offsets)


class OffsetNet(nn.Module):
    def __init__(self, in_channels, replace_stride_with_dilation):
        super(OffsetNet, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(BasicBlock(in_channels, 18)),
             nn.Sequential(nn.AvgPool2d(2),
                           BasicBlock(18, 18)),
             nn.Sequential(nn.AvgPool2d(2),
                           BasicBlock(18, 18)),
             nn.Sequential(nn.AvgPool2d(2) if not replace_stride_with_dilation else nn.Identity(),
                           BasicBlock(18, 18))
             ])

    def forward(self, x):
        x = x[0]
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features
