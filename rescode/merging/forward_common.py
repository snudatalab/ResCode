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

rescode/merging/forward_common.py
    - codes for the forward step shared by the RESCODE MoE blocks
"""
import torch


def add_original_cluster_output(final_hidden_states, hidden_states, member_ids,
                                selected_experts, routing_weights, original_experts):
    for j in member_ids:
        mask = selected_experts == j
        if not mask.any():
            continue
        top_x, idx = torch.where(mask)
        out = original_experts[j](hidden_states.index_select(0, top_x)) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, out.to(final_hidden_states.dtype))
