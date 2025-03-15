# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from llama_cookbook.data.concatenator import ConcatDataset
from llama_cookbook.my_datasets import DATASET_PREPROC, DATALOADER_COLLATE_FUNC
from llama_cookbook.utils.config_utils import get_dataloader_kwargs


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train",
    trait: str = None, tone: str = None, use_negative_essay: bool = False
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split # "src/llama_cookbook/datasets/personality_essay_train.json"
            if split == "train"
            else dataset_config.test_split # "src/llama_cookbook/datasets/personality_essay_validation.json"
        )
    return DATASET_PREPROC[dataset_config.dataset]( # get_custom_dataset
        dataset_config,
        tokenizer,
        get_split(),
        trait=trait,
        tone=tone,
        use_negative_essay=use_negative_essay,
    )

def get_custom_data_collator(
    dataset_processer, dataset_config
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATALOADER_COLLATE_FUNC:
        return None

    return DATALOADER_COLLATE_FUNC[dataset_config.dataset](
        dataset_processer,
        dataset_config
    )

def get_dataloader(tokenizer, dataset_config, train_config, split: str = "train"):
    dataset = get_preprocessed_dataset(tokenizer, dataset_config, split)
    dl_kwargs = get_dataloader_kwargs(train_config, dataset, tokenizer, split)
    
    if split == "train" and train_config.batching_strategy == "packing":
        dataset = ConcatDataset(dataset, chunk_size=train_config.context_length)

    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **dl_kwargs,
    )
    return dataloader
    