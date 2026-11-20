"""
ResCode: Accurate Expert Compression of Sparse MoE Models via Residual-code Restoration (EMNLP 2026)

Authors:
- Hyunwoo Yang (jg020247@snu.ac.kr), Seoul National University
- Minjun Kim (minjun.kim@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

Version : 1.0

Date : Nov. 20th, 2026

Main Contact: Hyunwoo Yang

This software is free of charge under research purposes.

For other purposes (e.g. commercial), please contact the authors.

rescode/utils/sparsity.py
    - codes for the stable rank of weights

This code is mainly based on
    - HC-SMoE: https://github.com/wazenmai/HC-SMoE
"""
import torch

from .constants import FP32_EPS


def compute_weight_stable_rank(weight: torch.Tensor) -> float:
    """
    Compute the stable rank of a weight matrix

    Parameters
    ----------
    weight: torch.Tensor
        The weight matrix

    Returns
    -------
    float
        The stable rank of the weight matrix

    Examples
    --------
    >>> import torch
    >>> compute_weight_stable_rank(torch.randn(100, 100)) > 20
    True
    """
    singular_values = torch.linalg.svdvals(weight)
    return torch.sum(singular_values ** 2).item() / (torch.max(singular_values).item() ** 2 + FP32_EPS)
