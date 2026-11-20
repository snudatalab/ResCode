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

rescode/utils/dummy_test.py
    - codes for dummy inputs for tests

This code is mainly based on
    - HC-SMoE: https://github.com/wazenmai/HC-SMoE
"""
import torch

__all__ = ['DUMMY_INPUT_IDS', 'DUMMY_LABELS']

DUMMY_INPUT_IDS = torch.randint(0, 100, (1, 128))
DUMMY_LABELS = torch.randint(0, 100, (1, 128))
