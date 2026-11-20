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

rescode/utils/helper.py
    - codes for GPU memory logging and JSON saving

This code is mainly based on
    - HC-SMoE: https://github.com/wazenmai/HC-SMoE
"""
import json
import torch.cuda as cuda


def print_gpu_memory():
    for i in range(cuda.device_count()):
        print(f"Device {i}: {cuda.memory_allocated(i) / 1024**2:.2f} MB")

def save_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4)
