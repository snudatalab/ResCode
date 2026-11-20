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

rescode/__init__.py
    - codes for package initialization

This code is mainly based on
    - HC-SMoE: https://github.com/wazenmai/HC-SMoE
"""
import os

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
