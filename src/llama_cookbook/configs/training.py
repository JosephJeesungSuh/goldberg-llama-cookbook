# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass


@dataclass
class train_config:
    model_name: str="meta-llama/Llama-2-7b-hf"
    model_nickname: str="llama2_7b_base"
    tokenizer_name: str=None
    enable_fsdp: bool=True # shards model parameters, optimizer states and gradients across DDP ranks
    low_cpu_fsdp: bool=True # saves cpu memory by loading pretrained model on rank0 only
    run_validation: bool=True
    batch_size_training: int=8
    batching_strategy: str="padding" #alternative: padding
    context_length: int=4096
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=100
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=4
    lr: float=1e-4
    weight_decay: float=0.0
    seed: int=42
    use_fp16: bool=False  # load model paramater in torch.float16 dtype (not recommended)
    mixed_precision: bool=True
    val_batch_size: int=8
    dataset = "personality_dataset" # "samsum_dataset"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=True # use parameter efficient fine tuning
    from_peft_checkpoint: str="" # if not empty and use_peft=True, will load the peft checkpoint and resume the fine-tuning on that checkpoint
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    freeze_LLM_only: bool = False # Freeze self-attention layers in the language_model. Vision model, multi_modal_projector, cross-attention will be fine-tuned
    quantization: str = None
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = True # Enable wandb for experient tracking
    save_metrics: bool = True # saves training metrics to a json file for later plotting
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler

    training_data_source: str = "human"
    which_scheduler: str = 'cosine'  # cosine or step
    warmup_ratio: float = 0.1
    gamma: float= 0.85 # multiplicatively decay the learning rate by gamma after each epoch

    # arguments for SFT finetuning (difference-of-logprob method)
    trait: str = 'SURGENCY'
    tone: str = 'positive'
    use_weighting: bool = False
    weighting_alpha: float = 1.0

    # arguments for regressor head training
    training_regression: bool = False
    training_regression_objective: str = "regression"  # regression or classification
    training_regression_target: str = "bigfive"  # bigfive or bfi2
    training_regressor_head: bool = False # if False, only training the LoRA module
    regressor_module_path: str = None # pre-trained regressor module checkpoint path
    
    regressor_lr: float = 1e-4
    regressor_weight_decay: float = 0.0
    regressor_p_dropout: float = 0.2 # dropout rate 
    regressor_l1_lambda: float = 0.0 # L1 regularization 
    regressor_layer_type: str = "mlp"
    regressor_layer_depth: int = 2 # depth of the regressor head
    regressor_hidden_dim_factor: str = "[1,4]" # hidden dimension factor for the regressor head
    
    hidden_layer_index: int = -1
    use_negative_essay: bool = True 
    add_stimulus: bool = True
    use_eos: bool = False