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

rescode/merging/utils.py
    - codes for random group labels of experts

This code is mainly based on
    - HC-SMoE: https://github.com/wazenmai/HC-SMoE
"""
import os
import random
from typing import Optional

import torch
from tqdm import tqdm


def generate_random_group_labels(
        num_experts: int,
        num_groups: int,
) -> torch.Tensor:
    """
    Assign random group labels to experts, with each group has at least one expert.

    Examples
    --------
    >>> generate_random_group_labels(10, 3).unique().sort()[0]
    tensor([0, 1, 2])
    """
    group_labels = torch.zeros(num_experts, dtype=torch.long)
    for i in range(num_groups):
        group_labels[i] = i
    for i in range(num_groups, num_experts):
        group_labels[i] = random.randint(0, num_groups - 1)
    return group_labels[torch.randperm(num_experts)]
