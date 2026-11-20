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

rescode/data/__init__.py
    - codes for importing data collators, evaluation functions, and preprocessors

This code is mainly based on
    - HC-SMoE: https://github.com/wazenmai/HC-SMoE
"""
from .data_collator import (
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling
)
from .evaluation import (
    get_classification_label_index_and_token_ids,
    get_evaluate_fn,
    EXTRA_KEYS_FOR_EVAL,
    gather_predictions_references_by_causal_lm_loss
)
from .preprocessors import (
    TASK_MAPPING_DATASET_ARGUMENTS,
    TASK_MAPPING_PROMPT_KEY,
    get_label_mapping_id,
    Seq2SeqDataPreProcessor,
    Seq2SeqZeroShotDataPreProcessor,
    CausalZeroShotDataPreProcessor,
    CausalOneShotDataPreProcessor,
    tokenize_seq2seq,
    tokenize_seq2seq_zero_shot,
    tokenize_seq2se2_to_causal_lm,
    tokenize_causal_generation,
    tokenize_causal_zero_shot,
    keep_only_supporting_facts_in_context_for_hotpotqa,
    build_index_for_dataset
)
