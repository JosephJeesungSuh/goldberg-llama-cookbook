# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib
from typing import Union, List, Dict


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import AutoTokenizer
import json
import numpy as np


from llama_cookbook.model_checkpointing import (
    save_fsdp_model_checkpoint_full,
    save_model_and_optimizer_sharded,
    save_optimizer_checkpoint,
    save_peft_checkpoint,
    save_model_checkpoint,
    save_regressor_checkpoint
)
from llama_cookbook.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_cookbook.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_cookbook.utils.flop_utils import FlopMeasure
def set_tokenizer_params(tokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank,warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


# class MultiValueClassifier(nn.Module):
#     """
#     This module performs multi-value classification:
#     - We have 60 discrete values (60 for BFI-2 classification)
#     - Each value can take 5 possible classes
#     - We flatten them into a single linear projection that outputs 60 * 5 = 300 logits
#     - Then we reshape [batch_size, 60, 5] for cross-entropy
#     """
#     def __init__(
#         self,
#         hidden_dim: int,
#         num_values: int = 60,
#         num_classes: int = 5,
#         layer_type: str = "linear",
#         depth: int = 2,
#         p_dropout: float = 0.2,
#     ):
#         super().__init__()
#         if layer_type == "linear":
#             self.linear = nn.Linear(hidden_dim, num_values * num_classes, dtype=torch.bfloat16)
#         elif layer_type == "mlp":
#             if depth == 2:
#                 self.linear = nn.Sequential(
#                     nn.Linear(hidden_dim, hidden_dim // 4, dtype=torch.bfloat16),
#                     nn.GELU(),
#                     nn.Dropout(p_dropout),
#                     nn.Linear(hidden_dim // 4, num_values * num_classes, dtype=torch.bfloat16),
#                 )
#             elif depth == 3:
#                 self.linear = nn.Sequential(
#                     nn.Linear(hidden_dim, hidden_dim // 2, dtype=torch.bfloat16),
#                     nn.GELU(),
#                     nn.Dropout(p_dropout),
#                     nn.Linear(hidden_dim // 2, hidden_dim // 8, dtype=torch.bfloat16),
#                     nn.GELU(),
#                     nn.Dropout(p_dropout),
#                     nn.Linear(hidden_dim // 8, num_values * num_classes, dtype=torch.bfloat16),
#                 )
#             else:
#                 raise ValueError(f"Invalid depth {depth} for MultiValueClassifier")
#         else:
#             raise ValueError(f"Invalid type {type} for MultiValueClassifier")
#         self.layer_type = layer_type
#         self.depth = depth
#         self.p_dropout = p_dropout
#         self.num_values = num_values
#         self.num_classes = num_classes

#     def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
#         logits = self.linear(hidden_state)
#         logits = logits.view(-1, self.num_values, self.num_classes)
#         return logits

class MultiValueClassifier(nn.Module):
    """
    This module performs multi-value classification:
    - We have 60 discrete values (60 for BFI-2 classification)
    - Each value can take 5 possible classes
    - We flatten them into a single linear projection that outputs 60 * 5 = 300 logits
    - Then we reshape [batch_size, 60, 5]
    """
    def __init__(
        self, hidden_dim: int, num_values: int, num_classes: int,
        layer_type: str, depth: int, p_dropout: float, dtype: torch.dtype,
    ):
        super().__init__()
        if layer_type == "linear":
            self.linear = nn.Linear(hidden_dim, num_values * num_classes, dtype=dtype)
        elif layer_type == "mlp":
            if depth == 2:
                self.linear = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4, dtype=dtype), nn.GELU(), nn.Dropout(p_dropout),
                    nn.Linear(hidden_dim // 4, num_values * num_classes, dtype=dtype),
                )
            elif depth == 3:
                self.linear = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2, dtype=dtype), nn.GELU(), nn.Dropout(p_dropout),
                    nn.Linear(hidden_dim // 2, hidden_dim // 8, dtype=dtype), nn.GELU(), nn.Dropout(p_dropout),
                    nn.Linear(hidden_dim // 8, num_values * num_classes, dtype=dtype),
                )
            elif depth == 4:
                self.linear = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2, dtype=dtype), nn.GELU(), nn.Dropout(p_dropout),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4, dtype=dtype), nn.GELU(), nn.Dropout(p_dropout),
                    nn.Linear(hidden_dim // 4, hidden_dim // 16, dtype=dtype), nn.GELU(), nn.Dropout(p_dropout),
                    nn.Linear(hidden_dim // 16, num_values * num_classes, dtype=dtype),
                )
            else:
                raise ValueError(f"Invalid depth {depth} for MultiValueClassifier")
        else:
            raise ValueError(f"Invalid type {layer_type} for MultiValueClassifier")
        self.layer_type = layer_type
        self.depth = depth
        self.p_dropout = p_dropout
        self.num_values = num_values
        self.num_classes = num_classes
        self.dtype = dtype

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        logits = self.linear(hidden_state)
        logits = logits.view(-1, self.num_values, self.num_classes)
        return logits


def load_TDA(
    tda_path: Union[str, Path],
    to_list: bool = True,
    seperate_tone: bool = False,
) -> Union[List[str], Dict[str, list[str]], Dict[str, Dict[str, List[str]]]]:
    """
    Loading the TDA json file with one the the three options.
    1) default: put all adjectives to a single list.
    2) to_list = False: return a dictionary, but combine positive / negative adjectives.
    3) to_list = False, seperate_tone = True: return a dictionary in original form with seperate tone.
    """
    try:
        tda_json = json.load(open(tda_path, "r"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"TDA file not found: {tda_path}") from exc
    # return TDA to a single list of all adjectives.
    # original .json file seperately stores positive and negative adjectives.
    if to_list:
        return [
            adj
            for _, wordlist in tda_json.items()
            for tone in ['positive', 'negative']
            for adj in wordlist[tone]
        ]
    # return a dictionary, but combine positive / negative adjectives.
    if not seperate_tone:
        return {
            factor: wordlist['positive'] + wordlist['negative']
            for factor, wordlist in tda_json.items()
        }
    # return a dictionary in original format
    return tda_json

PARENT_DIR = Path(__file__).resolve().parent.parent
tda_dict = load_TDA(
    tda_path = os.path.join(PARENT_DIR, "my_datasets/bfi2_60_description.json"),
    to_list = False,
    seperate_tone = True,
)
bfi_to_qid_map = json.load(open(os.path.join(PARENT_DIR, "my_datasets/bfi2_description_to_qid.json"), "r"))
traits = ['SURGENCY', 'AGREEABLENESS', 'CONSCIENTIOUSNESS', 'EMOTIONAL_STABILITY', 'INTELLECT']

def convert_bfi2_to_bfi(person_score):
    bfi_score = {}
    for trait in tda_dict.keys():
        bfi_score[trait] = []
        for tone in ['positive', 'negative']:
            for adj in tda_dict[trait][tone]:
                qid = bfi_to_qid_map[adj.strip()]
                score = person_score[qid-1]
                score = score if tone == 'positive' else -score
                bfi_score[trait].append(score)
    return {k: np.mean(v) for k, v in bfi_score.items()}


def train(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None, regressor_module=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predictions

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        print(f"Starting epoch {epoch}/{train_config.num_epochs}")
        print(f"train_config.max_train_step: {train_config.max_train_step}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            if regressor_module:
                regressor_module.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            elif torch.cuda.is_available():
                                batch[key] = batch[key].to('cuda:0')

                    with autocast():
                        if train_config.training_regression:
                            outputs = model(
                                input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                output_hidden_states=True,
                                return_dict=True
                            )
                            last_hidden = outputs.hidden_states[-1]
                            last_token_repr = last_hidden[:, -1 if train_config.use_eos else -2, :] # batch_size x hidden_dim
                            # logits = regressor_module(last_token_repr).squeeze(-1)
                            logits = regressor_module(last_token_repr).squeeze(1)
                            bigfive_scores = batch['bigfive_scores'].to(torch.bfloat16)
                            if train_config.trait is not 'all':
                                bigfive_scores = bigfive_scores[:,
                                    0 if train_config.trait == 'SURGENCY'
                                    else 1 if train_config.trait == 'AGREEABLENESS'
                                    else 2 if train_config.trait == 'CONSCIENTIOUSNESS'
                                    else 3 if train_config.trait == 'EMOTIONAL_STABILITY'
                                    else 4
                                ]
                                logits = logits.squeeze(-1)
                            loss = F.mse_loss(logits, bigfive_scores)

                        else:
                            if not train_config.use_weighting:
                                loss = model(**batch).loss
                            else:
                                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                                logits = outputs.logits
                                labels = batch['labels']
                                shifted_logits = logits[:, :-1, :].contiguous()
                                shifted_labels = labels[:, 1:].contiguous()
                                batch_size, seq_len, vocab_size = shifted_logits.shape
                                len_valid_labels_per_sample = torch.sum((shifted_labels != -100), dim=1)

                                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                                losses_per_token = loss_fct(
                                    shifted_logits.view(-1, vocab_size), shifted_labels.view(-1)
                                ).view(batch_size, seq_len)
                                mean_loss_per_sample = torch.sum(losses_per_token, dim=1) / len_valid_labels_per_sample

                                sample_weights = batch['trait_value_from_median']
                                sample_weights = 1 + sample_weights * train_config.weighting_alpha
                                weighted_loss_per_sample = sample_weights * mean_loss_per_sample
                                loss = torch.sum(weighted_loss_per_sample) / torch.sum(sample_weights)

                    total_loss += loss.detach().float()
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()
        should_save_model = train_config.save_model
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run, regressor_module)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
            should_save_model = train_config.save_model and eval_epoch_loss < best_val_loss
        
        checkpoint_start_time = time.perf_counter()
        if should_save_model:
            if train_config.enable_fsdp:
                dist.barrier()
            if train_config.use_peft:
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"we are about to save the PEFT modules")
                        if train_config.training_regression:
                            print(f"saving the PEFT modules with regressor")
                else:
                    print(f"we are about to save the PEFT modules")
                    if train_config.training_regression:
                        print(f"saving the PEFT modules with regressor")
                
                save_peft_checkpoint(model, train_config.output_dir)
                if train_config.training_regression:
                    if rank==0:
                        save_regressor_checkpoint(regressor_module, train_config.output_dir)

                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                else:
                    print(f"PEFT modules are saved in {train_config.output_dir} directory")

            else:
                if not train_config.enable_fsdp:
                    save_model_checkpoint(model, train_config.output_dir)
                    
                elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                    print(" Saving the FSDP model checkpoint using FULL_STATE_DICT")
                    print("=====================================================")
                    save_fsdp_model_checkpoint_full(
                        model, optimizer, rank, train_config, epoch=epoch
                    )
                    
                    if train_config.save_optimizer:
                        print(" Saving the FSDP optimizer using FULL_STATE_DICT")
                        print("=====================================================")
                        save_optimizer_checkpoint(
                            model, optimizer, rank, train_config, epoch=epoch
                        )
                    
                elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:

                    if train_config.save_optimizer:
                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                        print("=====================================================")
                        save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                    else:
                        print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                        print("=====================================================")
                        save_model_and_optimizer_sharded(model, rank, train_config)

                    
            if train_config.enable_fsdp:
                dist.barrier()
        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)

        if train_config.run_validation:
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(eval_epoch_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results

def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run, regressor_module=None):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    if regressor_module:
        regressor_module.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                if train_config.training_regression:
                    outputs = model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        output_hidden_states=True,
                        return_dict=True
                    )
                    last_hidden = outputs.hidden_states[-1]
                    last_token_repr = last_hidden[:, -1 if train_config.use_eos else -2, :] # batch_size x hidden_dim
                     # logits = regressor_module(last_token_repr).squeeze(-1)
                    logits = regressor_module(last_token_repr).squeeze(1)
                    bigfive_scores = batch['bigfive_scores'].to(torch.bfloat16)
                    if train_config.trait is not 'all':
                        bigfive_scores = bigfive_scores[:,
                            0 if train_config.trait == 'SURGENCY'
                            else 1 if train_config.trait == 'AGREEABLENESS'
                            else 2 if train_config.trait == 'CONSCIENTIOUSNESS'
                            else 3 if train_config.trait == 'EMOTIONAL_STABILITY'
                            else 4
                        ]
                        logits = logits.squeeze(-1)
                    loss = F.mse_loss(logits, bigfive_scores) * bigfive_scores.shape[0] / train_config.val_batch_size

                    # logits = regressor_module(last_token_repr)
                    # bfi2_labels = (2 + 2.0 * batch['bfi2_labels']).long()
                    # loss = F.cross_entropy(logits.view(-1, regressor_module.num_classes), bfi2_labels.view(-1))

                else:
                    if not train_config.use_weighting:
                        outputs = model(**batch)
                        loss = outputs.loss
                    else:
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels']
                        )
                        logits = outputs.logits
                        labels = batch['labels']
                        shifted_logits = logits[:, :-1, :].contiguous()
                        shifted_labels = labels[:, 1:].contiguous()
                        batch_size, seq_len, vocab_size = shifted_logits.shape
                        len_valid_labels_per_sample = torch.sum((shifted_labels != -100), dim=1)

                        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                        losses_per_token = loss_fct(
                            shifted_logits.view(-1, vocab_size), shifted_labels.view(-1)
                        ).view(batch_size, seq_len)
                        mean_loss_per_sample = torch.sum(losses_per_token, dim=1) / len_valid_labels_per_sample

                        sample_weights = batch['trait_value_from_median']
                        sample_weights = 1 + sample_weights * train_config.weighting_alpha
                        weighted_loss_per_sample = sample_weights * mean_loss_per_sample
                        loss = torch.sum(weighted_loss_per_sample) / torch.sum(sample_weights)
                    
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            )

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                    }, commit=False)

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False
                    
def freeze_LLM_only(model):
    """
    Freeze self-attention layers in the language_model. vision_model, multi_modal_projector, and cross-attention layers will be fine-tuned
    """
    for name, param in model.language_model.named_parameters():
                param.requires_grad = False
    for i, layer in enumerate(model.language_model.model.layers):
        if i in model.language_model.model.cross_attention_layers:
            for param in layer.parameters():
                param.requires_grad = True

def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only available in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")

def print_frozen_model_status(model, config, rank: int = 0) -> None:
    """
    Print the frozen status of the model's and the number of trainable parameters after frozen.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("After freezing the model:")
        print(f"--> {config.model_name} has {trainable_params / 1e6} Million trainable params\n")

        module_states = {}
        # Iterate over all parameters
        for name, param in model.named_parameters():
            # Extract the top-level module name (e.g., "vision_model", "language_model")
            top_module = name.split(".")[0]

            # Initialize a record for the top-level module
            if top_module not in module_states:
                module_states[top_module] = {"frozen": [], "unfrozen": []}

            # Group parameters into frozen or unfrozen
            if param.requires_grad:
                module_states[top_module]["unfrozen"].append(name)
            else:
                module_states[top_module]["frozen"].append(name)

        print("--> Model state after freezing:")
        # Analyze and print the results
        for module, states in module_states.items():
            frozen_params = states["frozen"]
            unfrozen_params = states["unfrozen"]

            if frozen_params and unfrozen_params:
                # Mixed state: both frozen and unfrozen parameters
                print(f"    {module}: Mixed")
            elif frozen_params:
                # All parameters are frozen
                print(f"    {module}: Frozen")
            else:
                # All parameters are unfrozen
                print(f"    {module}: Unfrozen")
        print("")


def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (following FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            print(f"training params are saved in {file_name}")

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
