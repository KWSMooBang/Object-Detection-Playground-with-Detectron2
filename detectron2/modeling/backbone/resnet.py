import numpy as np
import torch
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

from torch import nn

from detectron2.layers import (
    CNNBlockBase,
    Conv2d,
    ShapeSpec,
    get_norm,
)

from .backbone import Backbone
from .build import BACKBONE_REGISTRY

__all__ = [
    'BasicBlock',
    'ResNetBlockBase',
]

class BasicBlock(CNNBlockBase):
    """
    The basic residual block for ResNet-18 and ResNet-34
    """

    def __init__(self, in_channels, out_channels, stride=1, norm='BN'):
        """
        Args:
            in_channels (int): Number of input channles
            out_channels (int): Number of output channels
            stride (int): Stride for the first conv
            nrom (str or callable): normalization for all conv layers
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels)
            )
        else:
            self.shortcut = None
        
        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        self.conv2 = Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=get_norm(norm, out_channels)
        )

        for layer in [self.conv1, self.conv2, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def foward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        
        out += shortcut
        out = F.relu(out)
        return out

class BottleneckBlock(CNNBlockBase):
    """
    The standard bottleneck residual block used by ResNet-50, 101 and 152
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm='BN',
        stride_in_1x1=False,
        dilation=1
    ):
        """
        Args:
            bottleneck_channels (int): number of output channels for the 3x3 con layers
            num_groups (int): number of groups for the 3x3 conv layer
            norm (str or callable): normalization for all conv layers
            stride_in_1x1 (bool): when stride > 1, whether to put stride in the 
                first 1x1 convolution or the bottleneck 3x3 convolution
            dilation (int): the dilation rate of the 3x3 conv layer
        """
        super().__init__(in_channels, out_channels, stride)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                bottleneck_channels,
                kernel_size=1,
                stride=stride_in_1x1,
                bias=False,
                norm=get_norm(norm, bottleneck_channels)
            )
        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            stride=stride_3x3,
            padding=1 * dilation,
            bias=False,
            groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )

        self.conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        for layer in [self.conv1, self.conv2, self.conv3, self.shortcut]:
            if layer is not None:
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.conv3(out)
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        out += shortcut
        out = F.relu(out)
        return out

class BasicStem(CNNBlockBase):
    """
    The standard ResNet stem (layers before the first residual block)
    with a conv, relu, and max_pool
    """

    def __init__(self, in_channels=3, out_channels=64, norm='BN'):
        """
        Args:
            norm (str or callable): norm after the first conv layer
        """
        super().__init__(in_channels, out_channels, 2)
        self.in_channels = in_channels

        self.conv1 = Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
            norm=get_norm(norm, out_channels)
        )

        weight_init.c2_msra_fill(self.conv1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x

class ResNet(Backbone):
    def __init__(
        self, 
        stem, 
        stages, 
        num_classes=None, 
        out_features=None,
        freeze_at=0
    ):
        """
        Args:
            stem (nn.Module): a stem module
            stages (list[list[CNNBlockBase]]): several stages each contain multiple class
            num_classes (Nont or int): if None, will not perform classification
                                    otherwise, will create a linear layer
            out_features (list[str]): name of the layers whose outputs should 
                                    be returned in forward
            freeze_at (int): The number of stages at the beginning to freeze/
        """
        super().__init__()
        self.stem = stem
        self.num_classes = num_classes

        current_stride = self.stem.stride
        self._out_feature_strides = {'stem': current_stride}
        self._out_feature_channels = {'stem': self.stem.out_channels}

        self.stage_names, self.stages = [], []

        if out_features is not None:
            num_stages = max(
                [{'res2': 1, 'res3': 2, 'res4': 3, 'res5': 4}.get(f, 0)
                 for f in out_features]
            )
            stages = stages[:num_stages]
        for i, blocks in enumerate(stages):
            assert len(blocks) > 0, len(blocks)
            for block in blocks:
                assert isinstance(block, CNNBlockBase), block
            
            name = 'res' + str(i + 2)
            stage = nn.Sequential(*blocks)

            self.add_module(name, stage)
            self.stage_names.append(name)
            self.stages.append(stage)

            current_stride = int(
                current_stride * np.prod([b.stride for b in blocks])
            )
            current_channels = blocks[-1].out_channels
            self._out_feature_strides[name] = current_stride
            self._out_feature_channels[name] = current_channels

        self.stage_names = tuple(self.stage_names)

        if num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(current_channels, num_classes)

            nn.init.normal(self.linear.weight, std=0.01)
            name = 'linear'

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, f"Available children: {', '.join(children)}"
        self.freeze(freeze_at)
        

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N, C, H, W)
               H, W must be a mutliple of self.size_divisibility
        Returns:
            dict[str: Tensor]: names and teh corresponding features
        """
        assert x.dim == 4, f"ResNet takes an input of shape (N, C, H, W). Got {x.shape} instead."

        outputs = {}
        x = self.stem(x)
        if 'stem' in self._out_features:
            outputs['stem'] = x
        for name, stage in zip(self.stage_names, self.stages):
            x = stage(x)
            if name in self._out_feature:
                outputs[name] = x
            
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.linear(x)
            if 'linear' in self._out_features:
                outputs['linear'] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels = self._out_feature_channels[name],
                stride = self._out_feature_strides[name]
            )
            for name in self._out_features
        }
    
    def freeze(self, freeze_at=0):
        """
        Freeze the first several stages of the ResNet.
        Commonly used in fine-tuning

        Layers that produce the same feature map spatial size are defined
        as one 'stage'

        Args:
            freeze_at (int): number of stages to freeze
                '1': freezing the stem
                '2': freezing the stem and one residual stage
        
        Returns:
            nn.Module: this ResNet itself
        """
        if freeze_at >= 1:
            self.stem.freeze()
        for idx, stage in enumerate(self.stages, start=2):
            if freeze_at >= idx:
                for block in stage.children():
                    block.freeze()
        return self

    @staticmethod
    def make_stage(block_class, num_blocks, in_channels, out_channels, **kwargs):
        """
        Args:
            block_class (type): a subclass of CNNBlockBase that's used to create all blocks stage
            num_blocks (int): number of blocks in this stage
            in_channels (int): input channels of the entire stage
            out_channels (int): output channels of every block in the stage
            kwargs: other arguments

        Returns:
            list[CNNBlockBase]: a list of block module
        """
        blocks = []
        for i in range(num_blocks):
            curr_kwargs = {}
            for k, v in kwargs.items():
                if k.endswith('_per_block'):
                    assert len(v) == num_blocks, (
                        f"Argument '{k}' of make_stage should have the "
                        f"smae length as num_blocks={num_blocks}"
                    )
                    newk = k[: -len('_per_block')]
                    assert newk not in kwargs, f"Cannot call make_stage with both {k} and {newk}."
                    curr_kwargs[newk] = v[i]
                else:
                    curr_kwargs[k] = v
            blocks.append(
                block_class(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    **curr_kwargs
                )
            )
            in_channels = out_channels
        return blocks
    
    @staticmethod
    def make_default_stages(depth, block_class=None, **kwargs):
        """
        Created list of ResNet stages from pre-defined depth 
        (one of 18, 34, 50, 101, 152)
        Args:
            depth (int): depth of ResNet
            block_class (type): the CNN block class. Has to accept 'bottleneck_channels'
                                argument for depth > 50
            kwargs : other arguments 

        Returns:
            list[list[CNNBlockBase]]
        """
        num_blocks_per_stage = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }[depth]
        if block_class is None:
            block_class = BasicBlock if depth < 50 else BottleneckBlock
        if depth < 50:
            in_channels = [64, 64, 128, 256]
            out_channels = [64, 128, 256, 512]
        else: 
            in_channels = [64, 256, 512, 1024]
            out_channels = [256, 512, 1024, 2048]
        ret = []
        for n, s, i, o in zip(num_blocks_per_stage, [1, 2, 2, 2], in_channels, out_channels):
            if depth >= 50:
                kwargs['bottleneck_channels'] = o // 4
            ret.append(
                ResNet.make_stage(
                    block_class=block_class,
                    num_blocks=n,
                    stride_per_block=[s] + [1] * (n - 1),
                    in_channels=i,
                    out_channels=o,
                    **kwargs,
                )
            )
        return ret

ResNetBlockBase = CNNBlockBase

def make_stage(*args, **kwargs):
    return ResNet.make_stage(*args, **kwargs)

@BACKBONE_REGISTRY.register()
def build_resnet_backbone(cfg, input_shape):
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm
    )

    freeze_at           = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features        = cfg.MODEL.RESNETS.OUT_FEATURES
    depth               = cfg.MODEL.RESNETS.DEPTH
    num_groups          = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group     = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels         = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels        = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1       = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation       = cfg.MODEL.RESNETS.RES5_DILATION

    assert res5_dilation in {1, 2}, f"res5_dilation cannot be {res5_dilation}."

    num_blocks_per_stage = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3,4, 23, 3],
        152: [3, 8, 36, 3],
    }[depth]

    if depth in [18, 34]:
        assert out_channels == 64, "Must set MODEL.RESNETS.RES2_OUT_CHANNELS = 64 for R18/R34"
        assert res5_dilation == 1, "Must set MODEL.RESNETS.RES5_DILATION = 1 for R18/R34"
        assert num_groups == 1, "MUST set MODEL.RESNETS.NUM_GROUPS = 1 for R18/R34"
    
    stages = []

    for idx, stage_idx in enumerate(range(2, 6)):
        dilation = res5_dilation if stage_idx == 5 else 1
        first_stride = 1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
        stage_kargs = {
            'num_blocks': num_blocks_per_stage[idx],
            'stride_per_block': [first_stride] + [1] * (num_blocks_per_stage[idx] - 1),
            'in_channels': in_channels,
            'out_channels': out_channels,
            'norm': norm,
        }
        if depth in [18, 34]:
            stage_kargs['block_class'] = BasicBlock
        else:
            stage_kargs['bottleneck_channels'] = bottleneck_channels
            stage_kargs['stride_in_1x1'] = stride_in_1x1
            stage_kargs['dilation'] = dilation
            stage_kargs['num_groups'] = num_groups
            stage_kargs['block_class'] = BottleneckBlock
        
        blocks = ResNet.make_stage(**stage_kargs)
        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2
        stages.append(blocks)
    return ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)