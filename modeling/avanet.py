from argparse import ArgumentParser
import torch
from torch import nn
import warnings
from kornia.filters.sobel import SpatialGradient
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3PlusDecoder
from modeling.backbones.avanet_backbone import AvanetBackbone, AdaptedResnet
from modeling.reusable_blocks import conv1x1, conv3x3, SeparableConv2d, DeformableSeparableConv2d, Bottleneck, \
    BasicBlock
from modeling.flow_layer import FlowLayer, FlowAttention
from utils.utils import str2bool


class Avanet(nn.Module):
    def __init__(self, backbone='avanet',
                 decoder='avanet',
                 replace_stride_with_dilation=False,
                 no_blocks=(3, 3, 3, 2),
                 deformable=True,
                 px_per_iter=4,
                 grad_attention=True,
                 decoder_out_ch=512,
                 decoder_dspf_ch=(64, 128, 256),
                 decoder_rates=(4, 8, 12),
                 decoder_deformable=True,
                 ):
        super().__init__()
        self.backbone = backbone
        self.decoder_type = decoder
        self.spatial_grad = SpatialGradient()
        self.dem_bn = nn.BatchNorm2d(1)
        depth = 4

        grad_feats = 64
        self.grad_net = GradNet(grad_feats, replace_stride_with_dilation)

        if backbone == 'avanet':
            self.encoder = AvanetBackbone(replace_stride_with_dilation=replace_stride_with_dilation,
                                          no_blocks=no_blocks,
                                          deformable=deformable,
                                          groups=2)
        elif backbone == 'adapted_resnet18':
            self.encoder = AdaptedResnet(grad_feats,
                                         backbone='resnet18',
                                         replace_stride_with_dilation=replace_stride_with_dilation)
            depth = 5
        elif backbone == 'adapted_resnet34':
            self.encoder = AdaptedResnet(grad_feats,
                                         backbone='resnet34',
                                         replace_stride_with_dilation = replace_stride_with_dilation)
            depth = 5
        elif backbone == 'adapted_resnet50':
            self.encoder = AdaptedResnet(grad_feats,
                                         backbone='resnet50',
                                         replace_stride_with_dilation = replace_stride_with_dilation)
            depth = 5
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

        if decoder == 'avanet':
            self.decoder = AvanetDecoderOld(
                in_channels=self.encoder.out_channels,
                out_channels=256,
                depth=depth,
                px_per_iter=px_per_iter,
                grad_attention=grad_attention,
                replace_stride_with_dilation=replace_stride_with_dilation
            )
        elif decoder == 'avanet_new':
            self.decoder = AvanetDecoderNew(
                in_channels=self.encoder.out_channels,
                out_channels=decoder_out_ch,
                grad_feats=grad_feats,
                replace_stride_with_dilation=replace_stride_with_dilation,
                dspf_ch=decoder_dspf_ch,
                dil_rates=decoder_rates,
                pixels_per_iter=px_per_iter,
                deformable=decoder_deformable
            )
        elif decoder == 'deeplab':
            self.decoder = DeepLabV3PlusDecoder(
                self.encoder.out_channels,
                out_channels=256,
                atrous_rates=(12, 24, 36),
                output_stride=16,
            )
        else:
            raise NotImplementedError('decoder type not implemented')

        upsampling = 2 ** (depth - 3)
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
        dem_grads = self.spatial_grad(dem).squeeze(dim=1)
        dem = torch.sqrt(torch.square(dem_grads[:, [0], :, :]) + torch.square(dem_grads[:, [1], :, :]))
        dem = self.dem_bn(dem)

        grad_feats = self.grad_net(torch.cat([dem_grads, dem], dim=1))

        if self.backbone == 'avanet':
            x = self.encoder(x, dem_grads)
        elif 'adapted_resnet' in self.backbone:
            x = self.encoder(x, grad_feats)
        else:
            x = self.encoder(x)

        if self.decoder_type == 'avanet':
            x = self.decoder(x, dem_grads)
        elif self.decoder_type == 'avanet_new':
            x = self.decoder(x, dem_grads, grad_feats)
        else:
            x = self.decoder(*x)

        if self.training:
            x[0] = self.segmentation_head(x[0])
        else:
            x = self.segmentation_head(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        # allows adding model specific args via command line and logging them
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--decoder', type=str, default='avanet')
        parser.add_argument('--avanet_rep_stride_with_dil', type=str2bool, default='False',
                            help='Replace stride with dilation in backbone')
        parser.add_argument('--avanet_no_blocks', type=int, nargs='+', default=(3, 3, 3, 2))
        parser.add_argument('--avanet_deformable', type=str2bool, default='True', )
        parser.add_argument('--avanet_px_per_iter', type=int, default=4)
        parser.add_argument('--avanet_grad_attention', type=str2bool, default='True')
        parser.add_argument('--decoder_out_ch', type=int, default=512)
        parser.add_argument('--decoder_dspf_ch', type=int, nargs='+', default=(64, 128, 256))
        parser.add_argument('--decoder_rates', type=int, nargs='+', default=(4, 8, 12, 16))
        parser.add_argument('--decoder_deformable', type=str2bool, default=True)
        return parser


class GradNet(nn.Module):
    def __init__(self, no_features, replace_stride_with_dilation):
        super().__init__()
        self.blocks = nn.Sequential(nn.AvgPool2d(4),
                                    BasicBlock(3, 32),
                                    BasicBlock(32, no_features)
                                    )
        self.avgpool = nn.AvgPool2d(2)
        self.avgpool2 = nn.AvgPool2d(2) if not replace_stride_with_dilation else nn.Identity()

    def forward(self, x):
        out = []
        x = self.blocks(x)
        out.append(x)

        x = self.avgpool(x)
        out.append(x)

        x = self.avgpool(x)
        out.append(x)

        x = self.avgpool2(x)
        out.append(x)

        return out


class AvanetDecoderNew(nn.Module):
    def __init__(self, in_channels, out_channels, grad_feats, replace_stride_with_dilation,
                 dspf_ch=(64, 128, 256), dil_rates=(4, 8, 12), pixels_per_iter=4, deformable=True):
        super().__init__()

        self.out_channels = out_channels
        self.deformable = deformable
        self.replace_stride_with_dilation = replace_stride_with_dilation

        if deformable:
            self.dspfs = nn.ModuleList([
                DSPFDeform(in_channels[-3], dspf_ch[0], grad_feats, dil_rates, pixels_per_iter),
                DSPFDeform(in_channels[-2], dspf_ch[1], grad_feats, dil_rates, pixels_per_iter),
                DSPFDeform(in_channels[-1], dspf_ch[2], grad_feats, dil_rates, pixels_per_iter),
            ])
        else:
            self.dspfs = nn.ModuleList([
                DSPF(in_channels[-3], dspf_ch[0], dil_rates, pixels_per_iter),
                DSPF(in_channels[-2], dspf_ch[1], dil_rates, pixels_per_iter),
                DSPF(in_channels[-1], dspf_ch[2], dil_rates, pixels_per_iter),
            ])

        skip_ch = 48
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels[-4], skip_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_ch),
            nn.ReLU(),
        )

        self.downsample = nn.AvgPool2d(2)

        self.up_iters = (1, 2, 2) if replace_stride_with_dilation else (1, 2, 3)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        self.combine = nn.Sequential(
            SeparableConv2d(
                skip_ch + dspf_ch[0] + dspf_ch[1] + dspf_ch[2],
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv1x1 = nn.ModuleList([
            conv1x1(dspf_ch[0], 1),
            conv1x1(dspf_ch[1], 1),
            conv1x1(dspf_ch[2], 1)
        ])

    def forward(self, x, grads, grad_feats):
        # make gradient field magnitude independent
        for _ in range(3):
            grads = self.downsample(grads)
        grads = grads + 1e-6  # avoid dividing by zero
        grads = grads / grads.norm(p=None, dim=1, keepdim=True)
        grads2 = self.downsample(grads)
        grads4 = self.downsample(grads2) if not self.replace_stride_with_dilation else grads2
        grad_dir = [grads, grads2, grads4]

        res = []
        res.append(self.skip(x[-4]))

        outputs = []
        for i in range(3):
            if self.deformable:
                out = self.dspfs[i](x[-3+i], grad_dir[i], grad_feats[i+1])
            else:
                out = self.dspfs[i](x[-3+i], grad_dir[i])
            for _ in range(self.up_iters[i]):
                out = self.up(out)
            res.append(out)
            outputs.append(self.conv1x1[i](out))
        out = torch.cat(res, dim=1)
        out = self.combine(out)
        if self.training:
            outputs.insert(0, out)
            out = outputs
        return out


class AvanetDecoderOld(nn.Module):
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

        self.flows = nn.ModuleList([FlowLayer(in_channels[-1], 128, px_per_iter),
                                    FlowLayer(in_channels[-2], 64, px_per_iter),
                                    FlowLayer(in_channels[-3], 32, px_per_iter)])

        self.block = nn.ModuleList([Bottleneck(in_channels[-1] + 128, in_channels[-1]),
                                    Bottleneck(in_channels[-2] + 64 + in_channels[-1], in_channels[-2]),
                                    Bottleneck(in_channels[-3] + 32 + in_channels[-2], in_channels[-3])])

        self.skips = nn.ModuleList([Bottleneck(in_channels[-1], in_channels[-1]),
                                    Bottleneck(in_channels[-2], in_channels[-2]),
                                    Bottleneck(in_channels[-3], in_channels[-3])])

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


class DSPPDeform(nn.Module):
    def __init__(self, in_channels, out_channels, grad_feats, dilated_rates=(8, 16, 24, 32), pixels_per_iter=4):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        modules = []
        ConvModule = DSPFSeparableConvDeformable
        modules.append(ConvModule(in_channels, out_channels, grad_feats, dilated_rates[0]))
        modules.append(ConvModule(in_channels, out_channels, grad_feats, dilated_rates[1]))
        modules.append(ConvModule(in_channels, out_channels, grad_feats, dilated_rates[2]))
        modules.append(ConvModule(in_channels, out_channels, grad_feats, dilated_rates[3]))
        self.dilated_convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, grad_dir, grad_feats):
        res = []
        res.append(self.conv1x1(x))
        for conv in self.dilated_convs:
            res.append(conv(x, grad_feats))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DSPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilated_rates=(8, 16, 24, 32), pixels_per_iter=4):
        super().__init__()

        modules = []
        ConvModule = DSPFSeparableConv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        )

        modules.append(ConvModule(in_channels, out_channels, dilated_rates[0]))
        modules.append(ConvModule(in_channels, out_channels, dilated_rates[1]))
        modules.append(ConvModule(in_channels, out_channels, dilated_rates[2]))
        modules.append(ConvModule(in_channels, out_channels, dilated_rates[3]))
        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, grad_dir):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DSPFDeform(nn.Module):
    def __init__(self, in_channels, out_channels, grad_feats, dilated_rates=(8, 16, 24), pixels_per_iter=4):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        modules = []
        ConvModule = DSPFSeparableConvDeformable
        modules.append(ConvModule(in_channels, out_channels, grad_feats, dilated_rates[0]))
        modules.append(ConvModule(in_channels, out_channels, grad_feats, dilated_rates[1]))
        modules.append(ConvModule(in_channels, out_channels, grad_feats, dilated_rates[2]))
        self.dilated_convs = nn.ModuleList(modules)

        self.flow = FlowLayer(in_channels, out_channels, pixels_per_iter)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, grad_dir, grad_feats):
        res = []
        res.append(self.conv1x1(x))
        for conv in self.dilated_convs:
            res.append(conv(x, grad_feats))
        res.append(self.flow(x, grad_dir))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DSPF(nn.Module):
    def __init__(self, in_channels, out_channels, dilated_rates=(8, 16, 24), pixels_per_iter=4):
        super().__init__()

        modules = []
        ConvModule = DSPFSeparableConv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        )

        modules.append(ConvModule(in_channels, out_channels, dilated_rates[0]))
        modules.append(ConvModule(in_channels, out_channels, dilated_rates[1]))
        modules.append(ConvModule(in_channels, out_channels, dilated_rates[2]))
        self.convs = nn.ModuleList(modules)

        self.flow = FlowLayer(in_channels, out_channels, pixels_per_iter)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x, grad_dir):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res.append(self.flow(x, grad_dir))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DSPFSeparableConvDeformable(nn.Module):
    def __init__(self, in_channels, out_channels, grad_feats, dilation):
        super().__init__()
        self.offset_block = BasicBlock(grad_feats, 18)
        self.conv = DeformableSeparableConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, grad_feats):
        offsets = self.offset_block(grad_feats)
        x = self.conv(x, offsets)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DSPFSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            SeparableConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
