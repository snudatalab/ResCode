# -*- coding: utf-8 -*-
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

rescode/evaluation/__init__.py
    - codes for importing evaluation functions

This code is mainly based on
    - HC-SMoE: https://github.com/wazenmai/HC-SMoE
"""
# @Author: pingzhili
# @Time: 2024/2/11
from .lm_eval import evaluate_fewshot
from .minipile import evaluate_minipile_perplexity, get_minipile_dataloder, get_calib_dataloder
