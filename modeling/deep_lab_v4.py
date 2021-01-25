import torch
from typing import Optional
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
from segmentation_models_pytorch.encoders.resnet import ResNetEncoder
from torchvision.models.resnet import Bottleneck
from modeling.backbones.custom_resnet import resnet_standard, resnet_deformable, resnet_leaky, resnet_small



class DeepLabv4(SegmentationModel):
    """Adaptation of Deeplabv3+
    Args:
        encoder_name: name of classification model (without last dense layers) used as feature
                extractor to build segmentation model.
        encoder_depth: number of stages used in decoder, larger depth - more features are generated.
            e.g. for depth=3 encoder will generate list of features with following spatial shapes
            [(H,W), (H/2, W/2), (H/4, W/4), (H/8, W/8)], so in general the deepest feature will have
            spatial resolution (H/(2^depth), W/(2^depth)]
        encoder_weights: one of ``None`` (random initialization), ``imagenet`` (pre-training on ImageNet).
        encoder_output_stride: downsampling factor for deepest encoder features (see original paper for explanation)
        decoder_atrous_rates: dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels: a number of convolution filters in ASPP module (default 256).
        in_channels: number of input channels for model, default is 3.
        classes: a number of classes for output (output shape - ``(batch, classes, h, w)``).
        activation (str, callable): activation function used in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax2d``, callable, None]
        upsampling: optional, final upsampling factor
            (default is 8 to preserve input -> output spatial shape identity)
        aux_params: if specified model will have additional classification auxiliary output
            build on top of encoder, supported params:
                - classes (int): number of classes
                - pooling (str): one of 'max', 'avg'. Default is 'avg'.
                - dropout (float): dropout factor in [0, 1)
                - activation (str): activation function to apply "sigmoid"/"softmax" (could be None to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV4**
    """

    def __init__(
            self,
            encoder_name: str = 'resnet50',
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        if encoder_name == 'resnet50':
            self.encoder = ResNetEncoder(out_channels=(3, 64, 256, 512, 1024, 2048),
                                         block=Bottleneck,
                                         layers=[3, 4, 6, 3],
                                         depth=5,
                                         )
            self.encoder.set_in_channels(in_channels)
            self.encoder.make_dilated(
                stage_list=[5],
                dilation_list=[2]
            )
        elif encoder_name == 'avanet_standard':
            self.encoder = resnet_standard()
        elif encoder_name == 'avanet_deformable':
            self.encoder = resnet_deformable()
        elif encoder_name == 'avanet_leaky':
            self.encoder = resnet_leaky()
        elif encoder_name == 'avanet_small':
            self.encoder = resnet_small()
        else:
            raise NotImplementedError('No encoder found for: ' + encoder_name)

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=16,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None
