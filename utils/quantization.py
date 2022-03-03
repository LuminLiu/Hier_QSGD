import torch
import numpy as np

def randomk_nn(model, compression_ratio):
    if compression_ratio == 1.0:
        return model
    param = list(model.nn_layers.parameters())
    nc = len(param)
    for i in range(nc):
        param_shape = param[i].shape
        param[i].data = torch.flatten(param[i])
        if torch.cuda.is_available():
            mask = (1 / compression_ratio) * torch.ones(param[i].shape).cuda()
            indices = torch.randperm(mask.shape[0]).cuda()
        else:
            mask = (1 / compression_ratio) * torch.ones(param[i].shape)
            indices = torch.randperm(mask.shape[0])
        indices = indices[:int(indices.shape[0] * (1 - compression_ratio))]
        mask[indices] = 0
        param[i].data = torch.mul(param[i].data, mask)
        # param[i].data *= mask
        param[i].data = torch.reshape(param[i].data, param_shape)
    return model

def randomk_nne(model, compression_ratio):
    if compression_ratio == 1.0:
        return None
    param = list(model.parameters())
    nc = len(param)
    for i in range(nc):
        param_shape = param[i].shape
        param[i].data = torch.flatten(param[i])
        if torch.cuda.is_available():
            mask = (1 / compression_ratio) * torch.ones(param[i].shape).cuda()
            indices = torch.randperm(mask.shape[0]).cuda()
        else:
            mask = (1 / compression_ratio) * torch.ones(param[i].shape)
            indices = torch.randperm(mask.shape[0])
        indices = indices[:int(indices.shape[0] * (1-compression_ratio))]
        mask[indices] = 0
        param[i].data = torch.mul(param[i].data, mask)
        # param[i].data *= mask
        param[i].data = torch.reshape(param[i].data, param_shape)
    return None

"""
Quantization scheme for QSGD
Follows Alistarh, 2017 (https://arxiv.org/abs/1610.02132) but without the compression scheme.
"""

def sr(x, d):
    """quantize the tensor x in d level on the absolute value coef wise"""
    if torch.cuda.is_available():
        x = x.cpu()
    norm = torch.norm(x, p=2, dim=0)
    # in case norm is zero
    if norm == 0:
        return x.cuda() if torch.cuda.is_available() else x
    level_float = d * np.abs(x) / norm
    previous_level = np.floor(level_float)
    is_next_level = torch.tensor(np.random.rand(*x.shape)) < (level_float - previous_level)
    new_level = previous_level + is_next_level
    rounded_x = np.sign(x) * norm * new_level / d
    if torch.cuda.is_available():
        rounded_x = rounded_x.cuda()
    return rounded_x

def sr_nn(model, bits):
    param = list(model.nn_layers.parameters())
    nc = len(param)
    if bits == 32:
        # 32 means we will not quantize the update
        # but 32 is not the exact levels for floating point numbers stored
        return model
    d = 2**bits
    for i in range(nc):
        param_shape = param[i].shape
        param[i].data = torch.flatten(param[i])
        param[i].data = sr(param[i].data, d)
        param[i].data = torch.reshape(param[i].data, param_shape)
    return model

def sr_nne(model, bits):
    param = list(model.parameters())
    nc = len(param)
    if bits == 32:
        # 32 means we will not quantize the update
        # but 32 is not the exact levels for floating point numbers stored
        return None
    d = 2**bits
    for i in range(nc):
        param_shape = param[i].shape
        param[i].data = torch.flatten(param[i])
        param[i].data = sr(param[i].data, d)
        param[i].data = torch.reshape(param[i].data, param_shape)
    return None

def quantization_nn(model, compression_ratio, q_method):
    if q_method == 'sparsification':
        # for sparsification, compression_ratio = d/n
        return randomk_nn(model, compression_ratio)
    elif q_method == 'rounding':
        # for rounding, compression_ratio = bits, quantization level = 2^(bits)
        return sr_nn(model, compression_ratio)
    else: raise ValueError('This quantization method not impelmented:'+ q_method)

def quantization_nne(model, compression_ratio, q_method):
    if q_method == 'sparsification':
        randomk_nne(model, compression_ratio)
    elif q_method == 'rounding':
        sr_nne(model, compression_ratio)
    else: raise ValueError('This quantization method not impelmented:'+ q_method)
