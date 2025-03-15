# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class samsum_dataset:
    dataset: str =  "samsum_dataset"
    train_split: str = "train"
    test_split: str = "validation"


@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_cookbook/datasets/grammar_dataset/gtrain_10k.csv"
    test_split: str = "src/llama_cookbook/datasets/grammar_dataset/grammar_validation.csv"


@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_cookbook/datasets/alpaca_data.json"

@dataclass
class essay_dataset:
    dataset: str = "essay_dataset"
    file: str = "getting-started/finetuning/datasets/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"
    data_path: str = ""

@dataclass
class llamaguard_toxicchat_dataset:
    dataset: str = "llamaguard_toxicchat_dataset"
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class personality_dataset:
    dataset: str = "personality_dataset"
    file: str = "src/llama_cookbook/my_datasets/personality_dataset.py:get_preprocessed_personality_dataset"
    train_split: str = "src/llama_cookbook/my_datasets/personality_essay_train.json"
    test_split: str = "src/llama_cookbook/my_datasets/personality_essay_val.json"
    # pre_essay_question: str = "Question: Imagine you are about to move in with a new roommate whom you have not yet met and with whom you'll be living for the coming academic year. How would you describe your personality traits, your feelings, and your favorite activities to this person?"
    # essay_frame: str = "Answer:"
    pre_negative_essay_question: str = "Question: Research shows most people tend to express and describe themselves in a generally positive way, focusing on the desirable qualities or characteristics about themselves. Of course, we all have some negative characteristics and experience unpleasant feelings in our lives. What about yourself did you leave out in the previous writing task?"
    pre_essay_question: str = "Instruction: You should spend about 10 minutes describing your personality in your own words. Imagine you are about to move in with a new roommate whom you have not yet met and with whom you'll be living for the coming academic year. How would you describe your personality traits, your feelings, and your favorite activities to this person?\n\nYou now have 10 minutes to type in your description of your personality in the field below. Type quickly as the ideas come to you - correct spelling and grammar are not important in this exercise. Do try to describe your personality so that your future roommate could read this description and get to know who you really are."
    essay_frame: str = "Description:"