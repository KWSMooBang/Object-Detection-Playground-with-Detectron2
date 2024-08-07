
from torch import nn

from .batch_norm import FrozenBatchNorm2d
from .wrappers import Conv2d

"""
CNN building blocks
"""

class CNNBlockBase(nn.Module):
    """
    Attribute:
        in_channels (int)
        out_channels (int)
        stride (int)
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        
        return self