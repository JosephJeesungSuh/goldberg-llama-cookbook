# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


import ast
import dataclasses
from datetime import datetime
import os
import random
from collections import Counter
from warnings import warn, simplefilter

import fire
import numpy as np
import torch
import torch.optim as optim
from accelerate.utils import is_xpu_available

from llama_cookbook.configs import (
    fsdp_config as FSDP_CONFIG,
    quantization_config as QUANTIZATION_CONFIG,
    train_config as TRAIN_CONFIG,
)
from llama_cookbook.data.concatenator import ConcatDataset
from llama_cookbook.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

from llama_cookbook.utils import fsdp_auto_wrap_policy
from llama_cookbook.utils.config_utils import (
    check_fsdp_config,
    generate_dataset_config,
    generate_peft_config,
    get_dataloader_kwargs,
    update_config,
)
from llama_cookbook.utils.dataset_utils import (
    get_custom_data_collator,
    get_preprocessed_dataset,
)

from llama_cookbook.utils.fsdp_utils import hsdp_device_mesh, get_policies
from llama_cookbook.utils.train_utils import (
    clear_gpu_cache,
    freeze_transformer_layers,
    freeze_LLM_only,
    print_model_size,
    print_frozen_model_status,
    setup,
    setup_environ_flags,
    train,
    MultiValueClassifier
)
from peft import get_peft_model, PeftModel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR, LambdaLR
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    MistralForCausalLM,
    MllamaForConditionalGeneration,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.mllama.modeling_mllama import (
    MllamaCrossAttentionDecoderLayer,
    MllamaSelfAttentionDecoderLayer,
    MllamaVisionEncoderLayer,
)

simplefilter(action="ignore", category=FutureWarning)

def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_cookbook.configs import wandb_config as WANDB_CONFIG

    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run


def lr_lambda(current_step, warmup_steps, total_steps):
    """
    Cosine scheduler with warmup.
    Args:
        current_step: current step in the training loop
        warmup_steps: number of steps for warmup
        total_steps: total number of steps for training
    Returns:
        float: learning rate multiplier
    """
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress) * 3.14159265358979323846)))

def output_dir_formatter(train_config, **kwargs):
    """
    Format the output directory based on the training configuration.
    Args:
        base_dir: base directory for saving the model
        train_config: training configuration object
    """
    base_dir = train_config.output_dir
    if base_dir.endswith("/"):
        base_dir = base_dir[:-1]        
    base_dir += f"/data_source_{train_config.training_data_source}"

    if train_config.training_regression:
        base_dir += "/regression"
        base_dir += f"/use_negative_{train_config.use_negative_essay}_add_stimulus_{train_config.add_stimulus}_use_eos_{train_config.use_eos}_hidden_layer_idx_{train_config.hidden_layer_index}"
        base_dir += f"/{train_config.model_nickname}"
        base_dir += f"/objective_{train_config.training_regression_objective}"
        base_dir += f"/target_value_{train_config.training_regression_target}"
        base_dir += f"/train_regressor_head_{train_config.training_regressor_head}"

        if train_config.training_regressor_head is False and train_config.regressor_module_path is None:
            raise ValueError("Please provide a regressor module path if training_regressor_head is False.")
        renamed_regressor_module_path = train_config.regressor_module_path.replace("/", "--") if train_config.regressor_module_path is not None else "None"
        base_dir += f"/start_checkpoint_{renamed_regressor_module_path}"

        if train_config.training_regressor_head:
            base_dir += f"/regressor_layer_type_{train_config.regressor_layer_type}"
            if train_config.regressor_layer_type == "mlp":
                base_dir += f"/regressor_layer_depth_{train_config.regressor_layer_depth}"
                base_dir += f"/regressor_hidden_dim_factor_{train_config.regressor_hidden_dim_factor}"
                base_dir += f"/regressor_lr_{train_config.regressor_lr}_regressor_weight_decay_{train_config.regressor_weight_decay}_regressor_p_dropout_{train_config.regressor_p_dropout}_regressor_l1_lambda_{train_config.regressor_l1_lambda}"
            else:
                raise NotImplementedError("Only testing MLP regressor head")

        base_dir += f"/lr_{train_config.lr}_bs_{train_config.batch_size_training*int(os.environ.get('WORLD_SIZE', 1))}_wd_{train_config.weight_decay}"
        base_dir += f"/lora_r_{kwargs.get('lora_config.r', 8)}_lora_alpha_{kwargs.get('lora_config.lora_alpha', 32)}_lora_dropout_{kwargs.get('lora_config.lora_dropout', 0.05)}"

    else:
        base_dir += "/sft"
        base_dir += f"/use_negative_{train_config.use_negative_essay}"
        base_dir += f"/{train_config.model_nickname}"      
        base_dir += f"/{train_config.trait}_{train_config.tone}"
        base_dir += f"/use_weighting_{train_config.use_weighting}_alpha_{train_config.weighting_alpha}"
        base_dir += f"/lr_{train_config.lr}_bs_{train_config.batch_size_training*int(os.environ.get('WORLD_SIZE', 1))}_wd_{train_config.weight_decay}"
        base_dir += f"/lora_r_{kwargs.get('lora_config.r', 8)}_lora_alpha_{kwargs.get('lora_config.lora_alpha', 32)}_lora_dropout_{kwargs.get('lora_config.lora_dropout', 0.05)}"
    return base_dir


def main(**kwargs):
    print(f"--> main.py print all arguments: {kwargs}")
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    # lr_${LR}_bs_${BATCH_SIZE_TRAINING}/${REGRESSOR_LAYER_TYPE}_${REGRESSOR_LAYER_DEPTH}_${REGRESSOR_P_DROPOUT}_${WEIGHT_DECAY}/
    update_config((train_config, fsdp_config), **kwargs)
    print(f"--> main.py print all train configurations: {train_config}")
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    train_config.output_dir = output_dir_formatter(train_config, **kwargs)
    print(f"--> Output Directory: {train_config.output_dir}")

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
            formatted_time_tensor = torch.tensor([ord(c) for c in formatted_time], dtype=torch.int32, device="cuda")
        else:
            formatted_time_tensor = torch.empty(15, dtype=torch.int32, device="cuda")
        # synchronize the formatted time tensor across all ranks to ensure the output directory is consistent
        torch.distributed.barrier()
        torch.distributed.broadcast(formatted_time_tensor, src=0)
        torch.distributed.barrier()        
        formatted_time = ''.join([chr(c) for c in formatted_time_tensor.cpu().tolist() if c != 0])
    else:
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d_%H%M%S")    
    train_config.output_dir += "-" + formatted_time
    print(f"--> Output Directory (Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'}): {train_config.output_dir}")

    wandb_run = None
    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank == 0:
            wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)

    # setting quantization configs
    bnb_config = None
    if train_config.quantization:
        if type(train_config.quantization) == type(True):
            warn(
                "Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit' but this might change in the future.",
                FutureWarning,
            )
            train_config.quantization = "8bit"

        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError(
                "8bit quantization is not supported with FSDP, please use 4bit quantization"
            )

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    config = AutoConfig.from_pretrained(train_config.model_name)
    if config.model_type == "mllama":
        is_vision = True
        model = MllamaForConditionalGeneration.from_pretrained(
            train_config.model_name,
            quantization_config=bnb_config,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map=(
                "auto"
                if train_config.quantization and not train_config.enable_fsdp
                else None
            ),
            torch_dtype=torch.float16 if train_config.use_fp16 else "auto",
        )
        processor = AutoProcessor.from_pretrained(
            train_config.model_name
            if train_config.tokenizer_name is None
            else train_config.tokenizer_name
        )
        processor.tokenizer.padding_side = "right"
        model.supports_gradient_checkpointing = True
        model.language_model.supports_gradient_checkpointing = True
    elif config.model_type == "llama":
        is_vision = False
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            if rank == 0:
                model = LlamaForCausalLM.from_pretrained(
                    train_config.model_name,
                    quantization_config=bnb_config,
                    use_cache=use_cache,
                    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                    device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                    torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
                )
            else:
                llama_config = AutoConfig.from_pretrained(train_config.model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)
        else:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                quantization_config=bnb_config,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                torch_dtype=torch.float16 if train_config.use_fp16 else "auto",
            )
    elif config.model_type == "mistral":
        is_vision = False
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            if rank == 0:
                model = MistralForCausalLM.from_pretrained(
                    train_config.model_name,
                    quantization_config=bnb_config,
                    attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                    device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                    torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
                )
            else:
                mistral_config = AutoConfig.from_pretrained(train_config.model_name)
                mistral_config.use_cache = use_cache
                with torch.device("meta"):
                    model = MistralForCausalLM(mistral_config)
        else:
            model = MistralForCausalLM.from_pretrained(
                train_config.model_name,
                quantization_config=bnb_config,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
                device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
                torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
            )
    else:
        raise ValueError(
            f"Model type {config.model_type} is not supported. Please use llama or mllama model."
        )
    # Load the tokenizer and add special tokens
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name,
            add_prefix_space=False,
        )
    except:
        tokenizer = AutoTokenizer.from_pretrained(
            train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name,
        )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # If there is a mismatch between tokenizer vocab size and embedding matrix,
    # throw a warning and then expand the embedding matrix
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(
            "WARNING: Resizing the embedding matrix to match the tokenizer vocab size."
        )
        model.resize_token_embeddings(len(tokenizer))

    print("--> finetuning.py: printing model size..")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if (
        train_config.enable_fsdp
        and fsdp_config.pure_bf16
        and not train_config.quantization
    ):
        model.to(torch.bfloat16)

    if train_config.use_peft:
        # Load the pre-trained peft model checkpoint and setup its configuration
        if train_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(
                model, train_config.from_peft_checkpoint, is_trainable=True
            )
            peft_config = model.peft_config
        # Generate the peft config and start fine-tuning from original model
        else:
            peft_config = generate_peft_config(train_config, kwargs)
            model = get_peft_model(model, peft_config)
        if wandb_run:
            wandb_run.config.update(peft_config)
        model.print_trainable_parameters()

    hsdp_device_mesh_plan = None
    if (
        fsdp_config.hsdp
        and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD
    ):
        hsdp_device_mesh_plan = hsdp_device_mesh(
            replica_group_size=fsdp_config.replica_group_size,
            sharding_group_size=fsdp_config.sharding_group_size,
        )
        print("HSDP device mesh is ready")

    # setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        check_fsdp_config(fsdp_config)

        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)
            # print model size and frozen layers after freezing layers
            print_frozen_model_status(model, train_config, rank if train_config.enable_fsdp else 0)

        if not train_config.use_peft and train_config.freeze_LLM_only and config.model_type == "mllama":
            freeze_LLM_only(model)
            # print model size and frozen layers after freezing layers
            print_frozen_model_status(model, train_config, rank if train_config.enable_fsdp else 0)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        # Create the FSDP wrapper for MllamaSelfAttentionDecoderLayer,MllamaCrossAttentionDecoderLayer,MllamaVisionEncoderLayer in vision models
        if is_vision:
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(
                model,
                [
                    MllamaSelfAttentionDecoderLayer,
                    MllamaCrossAttentionDecoderLayer,
                    MllamaVisionEncoderLayer,
                ],
            )
        else:
            # Create the FSDP wrapper for LlamaDecoderLayer in text models
            if config.model_type == "mistral":
                my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [MistralDecoderLayer])
            else:
                my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [LlamaDecoderLayer])
        device_id = 0
        if is_xpu_available():
            device_id = torch.xpu.current_device()
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()

        regressor_module = None
        if train_config.training_regression:

            assert train_config.training_regression_objective in ["regression", "classification"]
            assert train_config.training_regression_target in ["bigfive", "bfi2"]
            assert train_config.regressor_layer_type in ["mlp", "linear"]
            regressor_hidden_dim_factor = train_config.regressor_hidden_dim_factor
            if not isinstance(regressor_hidden_dim_factor, list):
                regressor_hidden_dim_factor = ast.literal_eval(regressor_hidden_dim_factor)
            assert len(regressor_hidden_dim_factor) == train_config.regressor_layer_depth

            regressor_module = MultiValueClassifier(
                hidden_dim = model.lm_head.in_features,
                num_values = 1 if train_config.training_regression_objective == "regression" else 5,
                num_classes = 5 if train_config.training_regression_target == "bigfive" else 60,
                layer_type = train_config.regressor_layer_type,
                depth = train_config.regressor_layer_depth,
                p_dropout = train_config.regressor_p_dropout,
                dtype = torch.bfloat16,
                hidden_dim_factor = regressor_hidden_dim_factor,
            )
            if train_config.regressor_module_path is not None:
                regressor_module.load_state_dict(torch.load(train_config.regressor_module_path))
            regressor_module = regressor_module.to(local_rank)

        if train_config.freeze_LLM_only:
            use_orig_params = True
        else:
            use_orig_params = False
        model = FSDP(
            model,
            ignored_modules = [regressor_module] if train_config.training_regression else None,
            auto_wrap_policy=(my_auto_wrapping_policy if train_config.use_peft else wrapping_policy),
            cpu_offload=(
                CPUOffload(offload_params=True)
                if fsdp_config.fsdp_cpu_offload
                else None
            ),
            mixed_precision=(mixed_precision_policy if not fsdp_config.pure_bf16 else None),
            sharding_strategy=fsdp_config.sharding_strategy,
            device_mesh=hsdp_device_mesh_plan,
            device_id=device_id,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=(
                (
                    lambda module: module.to_empty(
                        device=torch.device("cuda"), recurse=False
                    )
                )
                if train_config.low_cpu_fsdp and rank != 0
                else None
            ),
            use_orig_params=use_orig_params,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        if is_xpu_available():
            model.to("xpu:0")
        elif torch.cuda.is_available():
            model.to("cuda")
    dataset_config = generate_dataset_config(train_config, kwargs)
    if is_vision:
        dataset_processer = processor
    else:
        dataset_processer = tokenizer

    # Load and preprocess the dataset for training and validation

    dataset_train = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="train",
        train_config=train_config,
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    dataset_val = get_preprocessed_dataset(
        dataset_processer,
        dataset_config,
        split="test",
        train_config=train_config,
    )
    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Validation Set Length = {len(dataset_val)}")

    if train_config.batching_strategy == "packing":
        if is_vision:
            raise ValueError("Packing is not supported for vision datasets")
        else:
            dataset_train = ConcatDataset(
                dataset_train, chunk_size=train_config.context_length
            )

    train_dl_kwargs = get_dataloader_kwargs(
        train_config, dataset_train, dataset_processer, "train"
    )
    custom_data_collator = get_custom_data_collator(dataset_processer, dataset_config)
    if custom_data_collator:
        print("custom_data_collator is used")
        train_dl_kwargs["collate_fn"] = custom_data_collator
    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )
    print(f"--> Num of Training Set Batches loaded = {len(train_dataloader)}")

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            if is_vision:
                raise ValueError("Packing is not supported for vision datasets")
            else:
                dataset_val = ConcatDataset(
                    dataset_val, chunk_size=train_config.context_length
                )

        val_dl_kwargs = get_dataloader_kwargs(
            train_config, dataset_val, dataset_processer, "val"
        )
        if custom_data_collator:
            val_dl_kwargs["collate_fn"] = custom_data_collator

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )
        print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")
        if len(eval_dataloader) == 0:
            raise ValueError(
                f"The eval set size is too small for dataloader to load even one batch. Please increase the size of eval set. ({len(eval_dataloader)=})"
            )
        else:
            print(f"--> Num of Validation Set Batches loaded = {len(eval_dataloader)}")

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
        optimizer_regressor_module = None
        if train_config.training_regression and train_config.training_regressor_head:
            optimizer_regressor_module = AnyPrecisionAdamW(
                regressor_module.parameters(),
                lr=train_config.regressor_lr,
                momentum_dtype=torch.bfloat16,
                variance_dtype=torch.bfloat16,
                use_kahan_summation=False,
                weight_decay=train_config.regressor_weight_decay,
            )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
        optimizer_regressor_module = None
        if train_config.training_regression and train_config.training_regressor_head:
                optimizer_regressor_module = optim.AdamW(
                regressor_module.parameters(),
                lr=train_config.regressor_lr,
                weight_decay=train_config.regressor_weight_decay,
            )
    # scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    if train_config.which_scheduler == "cosine":
        total_steps = int(
            len(train_dataloader)
            * train_config.num_epochs / train_config.gradient_accumulation_steps
        )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: lr_lambda(
                current_step = step,
                warmup_steps = int(train_config.warmup_ratio * float(total_steps)),
                total_steps = total_steps,
            )
        )
        scheduler_regressor_module = None
        if optimizer_regressor_module is not None:
            scheduler_regressor_module = LambdaLR(
                optimizer_regressor_module,
                lr_lambda=lambda step: lr_lambda(
                    current_step = step,
                    warmup_steps = int(train_config.warmup_ratio * float(total_steps)),
                    total_steps = total_steps,
                )
            )
        print(f"--> Using Cosine Scheduler with total_steps = {total_steps},"
               " warmup_steps = {int(train_config.warmup_ratio * total_steps)}")
    elif train_config.which_scheduler == 'step':
        scheduler = StepLR(
            optimizer,
            step_size=1,
            gamma=train_config.gamma ** (
                1.0 / float(len(train_dataloader))
                * float(train_config.gradient_accumulation_steps)
            ),
        )
        scheduler_regressor_module = None
        if optimizer_regressor_module is not None:
            scheduler_regressor_module = StepLR(
                optimizer_regressor_module,
                step_size=1,
                gamma=train_config.gamma ** (
                    1.0 / float(len(train_dataloader))
                    * float(train_config.gradient_accumulation_steps)
                ),
            )
        print(f"--> Using Step Scheduler with gamma = {scheduler.gamma}")

    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        wandb_run,
        regressor_module,
        optimizer_regressor_module,
        scheduler_regressor_module,
    )
    if not train_config.enable_fsdp or rank == 0:
        [print(f"Key: {k}, Value: {v}") for k, v in results.items()]
        if train_config.use_wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v


if __name__ == "__main__":
    fire.Fire(main)
