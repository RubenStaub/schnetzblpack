import numpy as np
import torch
from torch.nn import functional

__all__ = ["shifted_softplus", "softplus_inverse"]


def shifted_softplus(x):
    r"""Compute shifted soft-plus activation function.

    .. math::
       y = \ln\left(1 + e^{-x}\right) - \ln(2)

    Args:
        x (torch.Tensor): input tensor.

    Returns:
        torch.Tensor: shifted soft-plus of input.

    """
    return functional.softplus(x) - np.log(2.0)


def softplus_inverse(x):
    """Inverse softplus transformation. This is useful for initialization of parameters
    that are constrained to be positive.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))
