import torch
import torch.nn as nn
from torch.utils import model_zoo
from modeling.reusable_blocks import BasicBlock, SeparableConv2d
from torchvision.ops.deform_conv import DeformConv2d
from torchvision.models.resnet import ResNet
from segmentation_models_pytorch.encoders.resnet import resnet_encoders
from segmentation_models_pytorch.encoders import _utils as utils


class AdaptedResnet(ResNet):
    """ Version of the resnet that uses deformable convolutions in all its layers.

     Offsets for the deformable convolutions are calculated separately by OffsetNet.
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


class OffsetNet(nn.Module):
    """ Small network for computing the offsets to be used with the adapted resnet"""
    def __init__(self, in_channels, replace_stride_with_dilation):
        super(OffsetNet, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(BasicBlock(in_channels, in_channels),
                           SeparableConv2d(in_channels, 18, 3, padding=1)),
             nn.Sequential(nn.AvgPool2d(2),
                           BasicBlock(18, 36),
                           SeparableConv2d(36, 18, 3, padding=1)),
             nn.Sequential(nn.AvgPool2d(2),
                           BasicBlock(18, 36),
                           SeparableConv2d(36, 18, 3, padding=1)),
             nn.Sequential(nn.AvgPool2d(2) if not replace_stride_with_dilation else nn.Identity(),
                           BasicBlock(18, 36),
                           SeparableConv2d(36, 18, 3, padding=1))
             ])

    def forward(self, x):
        x = x[0]
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


class DeformableBasicBlock(nn.Module):
    """ Takes a standard basic block from the resnet and makes the first convolution deformable"""

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
    """ Slight modification on the DeformableBasicBlock which also adds squeeze and excitation to the block"""

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
