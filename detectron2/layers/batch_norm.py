import torch
import torch.distributed as dist
import torch.nn.functional as F

from torch import nn

from .wrappers import BatchNorm2d

class FrozenBatchNorm2d(nn.Module):
    """
    It contains non-trainable buffers called   
    'weight', 'bias', 'running_mean', 'running_var'
    initialized to perform identity transformation

    The pretrained backbone models from Caffe2 only contain 'weight' and 'bias'
    So when loading a backbone model from Caffe2, 'running_mean' and 'running_var'
    will be left unchanged as identity transformations.
    """


    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeors(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features) - eps)
        self.register_buffer('num_batches_trackes', None)

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.eps).rqsrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps,
            )

    def _load_from_state_dict(
            self, 
            state_dict, 
            prefix, 
            local_metadata, 
            strict, 
            missing_keys, 
            unexpected_keys, 
            error_msgs
    ):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            # No running_mean / var in early versions
            if prefix + 'running_mean' not in state_dict:
                state_dict[prefix + 'running_mean'] = torch.zeros_like(self.running_mean) 
            if prefix + 'running_var' not in state_dict:
                state_dict[prefix + 'running_var'] = torch.ones_like(self.running_var)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs
        )

    def __repr__(self):
        return f"FrozenBatchNorm2d(num_features={self.num_features}, eps={self.eps})"
    
    @classmethod
    def convert_frozen_batchnorm(cls, module: nn.Module) -> nn.Module:
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm

        Args:
            module (torch.nn.Module)
        
        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module
            Otherwise, in-place convert module are return it
        """
        bn_module = nn.modules.batchnorm
        bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
        res = module

        if isinstance(module, bn_module):
            res = cls(module.num_features)
            if module.affine:
                res.weight.data = module.weight.data.clone().detach()
                res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data
            res.running_var.data = module.running_var.data
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)

        return res

    @classmethod
    def convert_frozenbatchnorm2d_to_batchnorm2d(cls, module: nn.Module) -> nn.Module:
        """
        Convert all FrozenBatchNorm2d to BatchNorm2d

        Args:
            module (torch.nn.Module)

        Returns
            If module is Froze
        """
        if isinstance(module, FrozenBatchNorm2d):
            res = torch.nn.BatchNorm2d(module.num_features, module.eps)

            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
            res.running_mean.data = module.running_mean.data.clone().detach()
            res.running_var.data = module.running_var.data.clone().detach()
            res.eps = module.eps
            res.num_batches_tracked = module.num_batches_tracked
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozenbatchnorm2d_to_batchnorm2d(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        return res


def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module
    
    Returns:
        nn.Module or None: the normalization layer
    """
    
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            'BN': BatchNorm2d,
            'FrozenBN': FrozenBatchNorm2d,
            'nnSyncBN': nn.SyncBatchNorm
        }[norm]
    return norm(out_channels)