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

rescode/utils/__init__.py
    - codes for importing utility functions

This code is mainly based on
    - HC-SMoE: https://github.com/wazenmai/HC-SMoE
"""
from .constants import (
    FP32_EPS,
    BF16_EPS
)
from .dummy_test import (
    DUMMY_LABELS,
    DUMMY_INPUT_IDS
)
from .sparsity import (
    compute_weight_stable_rank
)
