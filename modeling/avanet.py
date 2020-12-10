from argparse import ArgumentParser
import torch
from torch import nn
import warnings
from kornia.filters.sobel import SpatialGradient
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from modeling.backbones.avanet_backbone import AvanetBackbone
from modeling.reusable_blocks import conv1x1, conv3x3, SeparableConv2d, Bottleneck
from utils.utils import str2bool


class Avanet(nn.Module):
    def __init__(self, backbone='avanet',
                 replace_stride_with_dilation=False,
                 no_blocks=(3, 3, 3, 2),
                 deformable=True,
                 px_per_iter=4,
                 grad_attention=True,
                 ):
        super().__init__()
        self.backbone = backbone
        self.spatial_grad = SpatialGradient()
        self.dem_bn = nn.BatchNorm2d(1)
        depth = 4

        if backbone == 'avanet':
            self.encoder = AvanetBackbone(replace_stride_with_dilation=replace_stride_with_dilation,
                                          no_blocks=no_blocks,
                                          deformable=deformable,
                                          groups=2)
        else:
            self.encoder = get_encoder(
                backbone,
                in_channels=3,
                depth=5,
                weights='imagenet',
            )
            if replace_stride_with_dilation:
                self.encoder.make_dilated(
                    stage_list=[5],
                    dilation_list=[2],
                )
            depth = 5

        self.decoder = AvanetDecoder(
            in_channels=self.encoder.out_channels,
            out_channels=256,
            depth=depth,
            px_per_iter=px_per_iter,
            grad_attention=grad_attention,
            replace_stride_with_dilation=replace_stride_with_dilation
        )

        upsampling = 2**(depth - 3)
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=1,
            activation=None,
            kernel_size=1,
            upsampling=upsampling,
        )

        self._init_weights()

    def _init_weights(self):
        warnings.warn('Avanet init weights function not implemented yet...')
        return

    def forward(self, x):
        # calculate dem gradients and magnitude
        dem = x[:, [-1], :, :]
        x = x[:, :-1, :, :]
        dem_grads = self.spatial_grad(dem).squeeze(dim=1)
        dem = torch.sqrt(torch.square(dem_grads[:, [0], :, :]) + torch.square(dem_grads[:, [1], :, :]))
        dem = self.dem_bn(dem)
        x = torch.cat([x, dem], dim=1)

        x = self.encoder(x, dem_grads) if self.backbone == 'avanet' else self.encoder(x)
        x = self.decoder(x, dem_grads)
        x = self.segmentation_head(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        # allows adding model specific args via command line and logging them
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--avanet_rep_stride_with_dil', type=str2bool, default='False',
                            help='Replace stride with dilation in backbone')
        parser.add_argument('--avanet_no_blocks', type=int, nargs='+', default=(3, 3, 3, 2))
        parser.add_argument('--avanet_deformable', type=str2bool, default='True',)
        parser.add_argument('--avanet_px_per_iter', type=int, default=4)
        parser.add_argument('--avanet_grad_attention', type=str2bool, default='True')
        return parser


class AvanetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4, px_per_iter=1, grad_attention=False,
                 replace_stride_with_dilation=False):
        super().__init__()
        self.out_channels = out_channels
        self.depth = depth
        self.attention = grad_attention
        self.replace_stride_with_dilation = replace_stride_with_dilation

        self.downsample = nn.AvgPool2d(2)
        if self.attention:
            self.grad_attention = FlowAttention(in_channels, replace_stride_with_dilation=replace_stride_with_dilation)

        self.flow1 = FlowLayer(in_channels[-1], 128, px_per_iter)
        self.flow2 = FlowLayer(in_channels[-2], 64, px_per_iter)
        self.flow3 = FlowLayer(in_channels[-3], 32, px_per_iter)
        self.flows = [self.flow1, self.flow2, self.flow3]

        self.block1 = Bottleneck(in_channels[-1] + 128, in_channels[-1])
        self.block2 = Bottleneck(in_channels[-2] + 64 + in_channels[-1], in_channels[-2])
        self.block3 = Bottleneck(in_channels[-3] + 32 + in_channels[-2], in_channels[-3])
        self.block = [self.block1, self.block2, self.block3]

        self.skip2 = Bottleneck(in_channels[-2], in_channels[-2])
        self.skip3 = Bottleneck(in_channels[-3], in_channels[-3])
        self.skips = [nn.Identity(), self.skip2, self.skip3]

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.final = nn.Sequential(
            SeparableConv2d(
                in_channels[-3],
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, grads):
        # make gradient field magnitude independent
        for _ in range(self.depth - 2):
            grads = self.downsample(grads)
        grads = grads + 1e-5  # avoid dividing by zero
        grads = grads / grads.norm(p=None, dim=1, keepdim=True)
        if self.attention:
            grads *= self.grad_attention(x)
        grads2 = self.downsample(grads)
        grads4 = self.downsample(grads2) if not self.replace_stride_with_dilation else grads2
        grads = [grads4, grads2, grads]

        high_features = None
        for i in range(3):
            features = self.skips[i](x[-i - 1])
            flow = self.flows[i](features, grads[i])
            tensors = [features, flow, high_features] if high_features is not None else [features, flow]
            features = torch.cat(tensors, dim=1)
            features = self.block[i](features)
            high_features = features if i == 0 and self.replace_stride_with_dilation else self.upsample(features)
        features = self.final(high_features)
        return features


class FlowLayer(nn.Module):
    """ Layer which implements flow propagation along a gradient field in both directions"""

    def __init__(self, inplanes, outplanes, pixels_per_iter=4):
        super().__init__()
        self.pixels_per_iter = pixels_per_iter
        self.conv1 = conv1x1(inplanes, outplanes)
        self.conv2 = conv1x1(inplanes, outplanes)
        self.sigmoid = nn.Sigmoid()
        self.merge = SeparableConv2d(2 * outplanes, outplanes, 3, padding=1)
        self.postprocess = SeparableConv2d(outplanes, outplanes, 3, padding=1)

    def forward(self, x, grads):
        iters = x.shape[2] // self.pixels_per_iter
        grads = grads / iters
        grads = grads.permute(0, 2, 3, 1).contiguous()

        m1 = self.conv1(x)
        m2 = self.conv2(x)
        m1 = self.sigmoid(m1)
        m2 = self.sigmoid(m2)
        for _ in range(iters):
            m1 = m1 + nn.functional.grid_sample(m1, grads)
            m2 = m2 + nn.functional.grid_sample(m2, -grads)
        m1 = self.sigmoid(m1)
        m2 = self.sigmoid(m2)
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
