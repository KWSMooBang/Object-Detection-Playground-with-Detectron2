import torch
import torch.nn.functional as F

from typing import List, Optional

class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around `torch.nn.Conv2d` to support 
    more features (normalization layer and activation layer)
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported

        Extra args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        """
        norm = kwargs.pop('norm', None)
        activation = kwargs.pop('activation', None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        x = F.conv2d(
            x, 
            self.weight, self.bias, 
            self.stride, self.padding,
            self.dilation, self.groups
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.acitvation(x)
            
        return x
    
ConvTranspose2d = torch.nn.ConvTranspose2d
BatchNorm2d = torch.nn.BatchNorm2d
interpolate = F.interpolate
Linear = torch.nn.Linear