# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import ast
import json
from multiprocessing import Lock

import datasets
import pandas as pd
import numpy as np

def get_preprocessed_personality_dataset(dataset_config, tokenizer, split, trait, tone, use_negative_essay):

    if trait not in ['SURGENCY', 'EMOTIONAL_STABILITY', 'AGREEABLENESS', 'INTELLECT', 'CONSCIENTIOUSNESS']:
        raise ValueError(f"--> get_preprocessed_personality_dataset: Invalid trait {trait}")
    if tone not in ['positive', 'negative','all']:
        raise ValueError(f"--> get_preprocessed_personality_dataset: Invalid tone {tone}")
    pre_essay_question = dataset_config.pre_essay_question
    pre_negative_essay_question = dataset_config.pre_negative_essay_question
    essay_frame = dataset_config.essay_frame

    model_name = tokenizer.name_or_path.split("/")[-1]
    preprocessed_file_dir = split.split(".")[0] + f"_model_{model_name}_{trait}_{tone}_preprocessed.json"
    if os.path.exists(preprocessed_file_dir):
        print(f"--> get_preprocessed_personality_dataset: preprocessed file exists.")
        with open(preprocessed_file_dir, 'r') as f:
            dataset = json.load(f)
            dataset = datasets.Dataset.from_dict(dataset)
        return dataset
    
    with open(split, 'r') as f:
        dataset = datasets.load_dataset('json', data_files=split, split='train')
    trait_median_value = np.median([x[trait] for x in dataset])
    if tone == 'positive':
        dataset = dataset.filter(lambda x: x[trait] >= trait_median_value)
    elif tone == 'negative':
        dataset = dataset.filter(lambda x: x[trait] < trait_median_value)
    elif tone == 'all':
        dataset = dataset
    print(f"--> get_preprocessed_personality_dataset: {len(dataset)} samples selected for {trait} {tone}.")


    def tokenize_add_label(sample):
        # prefix_essay = tokenizer.encode(
        #     tokenizer.bos_token + pre_essay_question.strip() + "\n\n" + essay_frame.strip(),
        #     add_special_tokens=False
        # )
        # essay = tokenizer.encode(
        #     (
        #         sample["text"].strip() # whitespace automatically added at the beginning
        #         + tokenizer.eos_token if not use_negative_essay else "\n\n"
        #     ),
        #     add_special_tokens=False
        # )
        if use_negative_essay:
            prefix_1 = (
                tokenizer.bos_token
                + pre_essay_question.strip()
                + "\n\n"
                + essay_frame.strip()
            )
            prefix_2 = (
                pre_negative_essay_question.strip()
                + "\n\n"
                + essay_frame.strip()
            )
            input_ids = tokenizer.encode(
                prefix_1
                + " " + sample["text"].strip() + "\n\n"
                + prefix_2
                + " " + sample["negative_text"].strip() + tokenizer.eos_token,
                add_special_tokens=False
            )
            len_prefix_1 = len(
                tokenizer.encode(prefix_1, add_special_tokens=False)
            )
            len_essay = len(
                tokenizer.encode(sample["text"].strip() + "\n\n", add_special_tokens=False)
            )
            len_prefix_2 = len(
                tokenizer.encode(prefix_2, add_special_tokens=False)
            )
            len_negative_essay = len(
                tokenizer.encode(sample["negative_text"].strip() + tokenizer.eos_token, add_special_tokens=False)
            )
            assert len(input_ids) == len_prefix_1 + len_essay + len_prefix_2 + len_negative_essay

        else:
            prefix = (
                tokenizer.bos_token
                + pre_essay_question.strip()
                + "\n\n"
                + essay_frame.strip()
            )
            input_ids = tokenizer.encode(
                prefix + " " + sample["text"].strip() + tokenizer.eos_token,
                add_special_tokens=False
            )
            len_prefix = len(
                tokenizer.encode(prefix, add_special_tokens=False)
            )
            len_essay = len(
                tokenizer.encode(sample["text"].strip() + tokenizer.eos_token,
                add_special_tokens=False)
            )
            assert len(input_ids) == len_prefix + len_essay

        sample_processed = {
            "input_ids": input_ids, # prefix_essay + essay,
            "attention_mask" : [1] * len(input_ids), # [1] * (len(prefix_essay) + len(essay)),
            "labels": (
                [-100] * len_prefix + input_ids[len_prefix:] if not use_negative_essay
                else (
                    [-100] * len_prefix_1
                    + input_ids[len_prefix_1: len_prefix_1 + len_essay]
                    + [-100] * len_prefix_2
                    + input_ids[len_prefix_1 + len_essay + len_prefix_2:]
                )
            ), # [-100] * len(prefix_essay) + essay,
            "trait_value_from_median": abs(sample[trait] - trait_median_value),
            "bfi_labels": sample['bfi_label'],
            }

        return sample_processed


    processed_dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    with open(preprocessed_file_dir, 'w') as f:
        json.dump(processed_dataset.to_dict(), f)

    return processed_dataset