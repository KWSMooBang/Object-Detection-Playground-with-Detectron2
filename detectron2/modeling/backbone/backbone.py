import torch.nn as nn
from abc import ABCMeta, abstractmethod
from typing import Dict

from detectron2.layers import ShapeSpec


__all__ = [
    'Backbone'
]

class Backbone(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self):
        pass

    @property
    def size_divisibility(self) -> int:
        """
        property for backbones which require the input height and width 
        to be divisible by a specific integer.
        Typically true for encoder / decoder type networks with lateral connection
        for which feature maps need to match dimension in the 'bottom up' and 'top down' paths.
        
        Returns:
            return 0 when if no specific input size deivisibility is required
        """
        return 0
    
    @property
    def padding_constraints(self) -> Dict[str, int]:
        """
        generalization of size_divisibility
        """
        return {}
    
    def output_shape(self):
        """
        Returns:
            dict[str -> ShapeSpec]
        """
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }