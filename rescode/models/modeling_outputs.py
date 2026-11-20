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

rescode/models/modeling_outputs.py
    - codes for output classes of MoE models

This code is mainly based on
    - HC-SMoE: https://github.com/wazenmai/HC-SMoE
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput

__all__ = [
    "MoEModelOutputWithPastAndCrossAttentions",
    "MoECausalLMOutputWithPastAndCrossAttentions",
    "MoECausalLMDistillationOutputWithPastAndCrossAttention",
    "MoEDistillationOutput",
]


@dataclass
class MoEModelOutputWithPastAndCrossAttentions(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    router_probs: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMOutputWithPastAndCrossAttentions(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MoECausalLMDistillationOutputWithPastAndCrossAttention(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None
    kd_loss: torch.FloatTensor = None
    lm_loss: torch.FloatTensor = None


@dataclass
class MoEDistillationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    kd_loss: torch.FloatTensor = None
    task_loss: torch.FloatTensor = None
    hd_loss: torch.FloatTensor = None
    norm_loss: torch.FloatTensor = None
    sing_reg_loss: torch.FloatTensor = None
    orth_reg_loss: torch.FloatTensor = None
    student_logits: torch.FloatTensor = None
    teacher_logits: torch.FloatTensor = None
