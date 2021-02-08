import torch
from torch import nn
from argparse import ArgumentParser
from kornia.filters.sobel import SpatialGradient
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3PlusDecoder
from modeling.avanet_backbone import AdaptedResnet
from modeling.reusable_blocks import conv1x1, SeparableConv2d, DeformableSeparableConv2d, BasicBlock
from modeling.flow_layer import FlowLayer
from utils.utils import str2bool


class Avanet(nn.Module):
    """
    Architecture based on Deeplabv3+ modified for avalanche mapping.

    The main differences are a deformable backbone with offsets computed by a smaller offsetnet, and an improved decoder
    which takes features from all resnet layers.

    The model is configurable to replace encoder and decoder parts with those from deeplabv3+.

    :param in_channels: the number of input channels to the network
    :param dem: whether the last input channel is a DEM
    :param backbone: which backbone to use. Can be adapted_resnetxx, resnetxx or any other model from
                    pytorch_segmentation_models submodule
    :param replace_stride_with_dilation: whether to replace the stride in the last backbone layer with a dilated convolution
    :param px_per_iter: the number of pixels per iteration to be used in the flow layers
    :param decoder_out_ch: the number of output channels from the decoder
    :param decoder_dspf_ch: the number of output channels of each DSPF modules given as a list for each backbone layer
    :param decoder_rates: list of dilation rates to be used in the DSPF module

    :returns: Predicted logits for each pixel. If in training mode, returns a list with intermediate predictions as well
              Predicted logits can be passed through a sigmoid to give the likelihood of an avalanche.
    """
    def __init__(self, in_channels=3,
                 dem=True,
                 backbone='adapted_resnet34',
                 decoder='avanet',
                 replace_stride_with_dilation=True,
                 px_per_iter=4,
                 decoder_out_ch=512,
                 decoder_dspf_ch=(64, 128, 256),
                 decoder_rates=(4, 8, 12),
                 ):
        super().__init__()

        # handle 'avanet_new' decoder in case checkpoint from older code is used
        if decoder == 'avanet_new':
            decoder = 'avanet'

        self.backbone = backbone
        self.decoder_type = decoder
        self.spatial_grad = SpatialGradient()
        self.dem_bn = nn.BatchNorm2d(1)

        dem_feats = 64
        self.grad_net = DEMFeatureNet(dem_feats, replace_stride_with_dilation)

        if 'adapted_resnet' in backbone:
            self.encoder = AdaptedResnet(dem_feats,
                                         in_channels=in_channels,
                                         dem=dem,
                                         backbone=backbone.replace('adapted_', ''),
                                         replace_stride_with_dilation=replace_stride_with_dilation)
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

        if decoder == 'avanet':
            self.decoder = AvanetDecoder(
                in_channels=self.encoder.out_channels,
                out_channels=decoder_out_ch,
                dem_feats=dem_feats,
                replace_stride_with_dilation=replace_stride_with_dilation,
                dspf_ch=decoder_dspf_ch,
                dil_rates=decoder_rates,
                pixels_per_iter=px_per_iter,
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

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=1,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

    def forward(self, x):
        # calculate DEM gradients and magnitude
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
            x = self.decoder(x, dem_grads, grad_feats)
        else:
            x = self.decoder(*x)

        # return all intermediate predictions for deep supervision, if in training mode
        if self.training and isinstance(x, list):
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
        parser.add_argument('--avanet_px_per_iter', type=int, default=4)
        parser.add_argument('--decoder_out_ch', type=int, default=512)
        parser.add_argument('--decoder_dspf_ch', type=int, nargs='+', default=(64, 128, 256))
        parser.add_argument('--decoder_rates', type=int, nargs='+', default=(4, 8, 12, 16))

        return parser


class DEMFeatureNet(nn.Module):
    """ This small network computes some general features from the DEM to be reused by the backbone and decoder
    Features are returned at each resolution of the backbone in a list

    :param no_features: number of input features
    :param replace_stride_with_dilation: whether to replace the stride in the last backbone layer with a dilated convolution
    :returns: a list of features corresponding to the spatial resolution of each backbone layer
    """

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


class AvanetDecoder(nn.Module):
    """ The avanet decoder takes features from all backbone layers and processes each with a DSPF module before adding
    them together with a concatenation and convolution.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param dem_feats: number of feature channels from the DEMFeatureNet
    :param replace_stride_with_dilation: whether to replace the stride in the last backbone layer with a dilated convolution
    :param dspf_ch: the number of output channels of each DSPF modules given as a list for each backbone layer
    :param dil_rates: list of dilation rates to be used in the DSPF module
    :param pixels_per_iter: the number of pixels per iteration to be used in the flow layers
    """

    def __init__(self, in_channels, out_channels, dem_feats, replace_stride_with_dilation,
                 dspf_ch=(64, 128, 256), dil_rates=(4, 8, 12), pixels_per_iter=4):
        super().__init__()

        self.out_channels = out_channels
        self.replace_stride_with_dilation = replace_stride_with_dilation

        self.dspfs = nn.ModuleList([
            DSPF(in_channels[-3], dspf_ch[0], dem_feats, dil_rates, pixels_per_iter),
            DSPF(in_channels[-2], dspf_ch[1], dem_feats, dil_rates, pixels_per_iter),
            DSPF(in_channels[-1], dspf_ch[2], dem_feats, dil_rates, pixels_per_iter),
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
        # make gradient field magnitude independent such that it becomes a vector field describing the downhill direction
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
            out = self.dspfs[i](x[-3+i], grad_dir[i], grad_feats[i+1])
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


class DSPF(nn.Module):
    """ The DSPF (Deformable Spatial Pyramid Flow) module is similar to the ASPP module from deeplabv3+. Multiple
    convolutional operations are performed with different dilations. Additionally, a flow layer is added.

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param dem_feats: number of feature channels from the DEMFeatureNet
    :param dilated_rates: list of dilation rates to be used
    :param pixels_per_iter: the number of pixels per iteration to be used in the flow layer
    """

    def __init__(self, in_channels, out_channels, dem_feats, dilated_rates=(8, 16, 24), pixels_per_iter=4):
        super().__init__()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        modules = []
        modules.append(DSPFSeparableConv(in_channels, out_channels, dem_feats, dilated_rates[0]))
        modules.append(DSPFSeparableConv(in_channels, out_channels, dem_feats, dilated_rates[1]))
        modules.append(DSPFSeparableConv(in_channels, out_channels, dem_feats, dilated_rates[2]))
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


class DSPFSeparableConv(nn.Module):
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
