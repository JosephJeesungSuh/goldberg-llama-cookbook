# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import ast
import json
from multiprocessing import Lock

import datasets
import pandas as pd
import numpy as np

def get_preprocessed_personality_dataset(dataset_config, tokenizer, split, train_config):

    trait = train_config.trait
    tone = train_config.tone
    use_negative_essay = train_config.use_negative_essay
    training_regression = train_config.training_regression
    add_stimulus = train_config.add_stimulus
    
    pre_essay_question = dataset_config.pre_essay_question
    pre_negative_essay_question = dataset_config.pre_negative_essay_question
    essay_frame = dataset_config.essay_frame
    stimulus = dataset_config.stimulus

    if trait not in ['SURGENCY', 'EMOTIONAL_STABILITY', 'AGREEABLENESS', 'INTELLECT', 'CONSCIENTIOUSNESS', 'all']:
        raise ValueError(f"--> get_preprocessed_personality_dataset: Invalid trait {trait}")
    if tone not in ['positive', 'negative','all']:
        raise ValueError(f"--> get_preprocessed_personality_dataset: Invalid tone {tone}")

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
    if tone == 'positive':
        trait_median_value = np.median([x[trait] for x in dataset])
        dataset = dataset.filter(lambda x: x[trait] >= trait_median_value)
    elif tone == 'negative':
        trait_median_value = np.median([x[trait] for x in dataset])
        dataset = dataset.filter(lambda x: x[trait] < trait_median_value)
    elif tone == 'all':
        dataset = dataset
    print(f"--> get_preprocessed_personality_dataset: {len(dataset)} samples selected for {trait} {tone}.")


    def tokenize_add_label(sample):
        
        if use_negative_essay:
            prefix_1 = (
                tokenizer.bos_token + " "
                + pre_essay_question.strip()
                + "\n\n"
                + essay_frame.strip()
            )
            essay_1 = " " + sample["text"].strip() + "\n\n"
            prefix_2 = (
                pre_negative_essay_question.strip()
                + "\n\n"
                + essay_frame.strip()
            )
            essay_2 = " " + sample["negative_text"].strip()
            input_ids = tokenizer.encode(
                prefix_1 + essay_1 + prefix_2 + essay_2
                + (stimulus if add_stimulus else "") + tokenizer.eos_token,
                add_special_tokens=False
            )
            len_prefix_1 = len(tokenizer.encode(prefix_1, add_special_tokens=False))
            len_essay = len(tokenizer.encode(essay_1, add_special_tokens=False))
            len_prefix_2 = len(tokenizer.encode(prefix_2, add_special_tokens=False))
            len_negative_essay = len(tokenizer.encode(essay_2, add_special_tokens=False))
            assert len(input_ids) == (
                len_prefix_1 + len_essay + len_prefix_2 + len_negative_essay
                + (len(tokenizer.encode(stimulus, add_special_tokens=False)) if add_stimulus else 0) + 1
            )

        else:
            prefix = (
                tokenizer.bos_token + " "
                + pre_essay_question.strip()
                + "\n\n"
                + essay_frame.strip()
            )
            essay = " " + sample["text"].strip()
            input_ids = tokenizer.encode(
                prefix + essay
                + (stimulus if add_stimulus else "") + tokenizer.eos_token,
                add_special_tokens=False
            )
            len_prefix = len(tokenizer.encode(prefix, add_special_tokens=False))
            len_essay = len(tokenizer.encode(essay, add_special_tokens=False))
            assert len(input_ids) == (
                len_prefix + len_essay
                + (len(tokenizer.encode(stimulus, add_special_tokens=False)) if add_stimulus else 0) + 1
            )

        sample_processed = {
            "input_ids": input_ids,
            "attention_mask" : [1] * len(input_ids),
            "labels": (
                (
                    [-100] * len_prefix
                    + input_ids[len_prefix:len_prefix + len_essay]
                    + [-100] * (len(input_ids) - len_prefix - len_essay)
                ) if not use_negative_essay
                else (
                    [-100] * len_prefix_1
                    + input_ids[len_prefix_1: len_prefix_1 + len_essay]
                    + [-100] * len_prefix_2
                    + input_ids[len_prefix_1 + len_essay + len_prefix_2: len_prefix_1 + len_essay + len_prefix_2 + len_negative_essay]
                    + [-100] * (len(input_ids) - len_prefix_1 - len_essay - len_prefix_2 - len_negative_essay)
                )
            ),
            "trait_value_from_median": (abs(sample[trait] - trait_median_value)) if (trait != 'all' and tone != 'all') else 0,
            "bfi2_labels": sample['bfi_label'],
            "bigfive_scores": [
                sample['SURGENCY'],
                sample['AGREEABLENESS'],
                sample['CONSCIENTIOUSNESS'],
                sample['EMOTIONAL_STABILITY'],
                sample['INTELLECT'],
            ]
        }

        return sample_processed

    processed_dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))
    with open(preprocessed_file_dir, 'w') as f:
        json.dump(processed_dataset.to_dict(), f)

    return processed_dataset