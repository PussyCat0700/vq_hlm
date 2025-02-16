#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, List

import datasets
import evaluate
import torch
from datasets import load_dataset
import light_hf_proxy

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.42.0.dev0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

padding_index = -100

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    vqvae_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The Wav2vec2 model checkpoint for weights initialization."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    feature_extractor_layers: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "feature_extractor_layers"
            )
        },
    )
    train_layer_id: Optional[int] = field(
        default=12,
        metadata={
            "help": (
                "train_layer_id"
            )
        },
    )
    # feature_extractor_layers: List[int] = field(
    #     default_factory=lambda: [5, 3]
    # )
    num_hidden_layers: Optional[int] = field(
        default=12,
        metadata={
            "help": (
                "num_hidden_layers"
            )
        },
    )
    decoder_layers: Optional[int] = field(
        default=4,
        metadata={
            "help": (
                "decoder_layers"
            )
        },
    )
    w_size: Optional[int] = field(
        default=32,
        metadata={
            "help": (
                "w_size"
            )
        },
    )
    quan_num: Optional[int] = field(
        default=8,
        metadata={
            "help": (
                "quan_num"
            )
        },
    )
    zquant_dim: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "zquant_dim"
            )
        },
    )
    n_embeddings: Optional[int] = field(
        default=256,
        metadata={
            "help": (
                "n_embeddings"
            )
        },
    )
    enc_layer: Optional[int] = field(
        default=6,
        metadata={
            "help": (
                "enc_layer"
            )
        },
    )
    dec_layer: Optional[int] = field(
        default=6,
        metadata={
            "help": (
                "dec_layer"
            )
        },
    )
    use_vqvae: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "use_vqvae"
            )
        },
    )
    wandb_project: Optional[str] = field(
        default='',
        metadata={
            "help": (
                "wandb_project"
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    eval_subset: str = field(default='validation')
    stride: int = field(default=512)

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    import os
    os.environ["WANDB_PROJECT"]=model_args.wandb_project
    
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                **dataset_args,
            )
    if not (training_args.do_train or data_args.eval_subset == 'train'):
        # If not training and not evaluating on train, we do not need to process it
        del raw_datasets["train"]
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # tokenizer.pad_token = tokenizer.eos_token

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        gpt_configs = AutoConfig.from_pretrained(model_args.model_name_or_path) 

        gpt_configs.model_name_or_path = getattr(gpt_configs, "model_name_or_path", model_args.model_name_or_path)
        gpt_configs.feature_extractor_layers = getattr(gpt_configs, "feature_extractor_layers", model_args.feature_extractor_layers)
        gpt_configs.decoder_layers = getattr(gpt_configs, "decoder_layers", model_args.decoder_layers)
        gpt_configs.w_size = getattr(gpt_configs, "w_size", model_args.w_size)
        gpt_configs.train_layer_id = getattr(gpt_configs, "train_layer_id", model_args.train_layer_id)
        gpt_configs.use_vqvae = getattr(gpt_configs, "use_vqvae", model_args.use_vqvae)

        if model_args.use_vqvae:
            gpt_configs.vqvae_model_path = getattr(gpt_configs, "vqvae_model_path", model_args.vqvae_model_path)
            gpt_configs.n_embeddings = getattr(gpt_configs, "n_embeddings", model_args.n_embeddings)
            gpt_configs.quan_num = getattr(gpt_configs, "quan_num", model_args.quan_num)
            gpt_configs.zquant_dim = getattr(gpt_configs, "zquant_dim", model_args.zquant_dim)
            gpt_configs.enc_layer = getattr(gpt_configs, "enc_layer", model_args.enc_layer)
            gpt_configs.dec_layer = getattr(gpt_configs, "dec_layer", model_args.dec_layer)
            from models.context_decoder.context_decoder_vqvae import GPT2LMHeadModel
        else:
            from models.context_decoder.context_decoder_mse import GPT2LMHeadModel

        assert (model_args.feature_extractor_layers + model_args.decoder_layers) == gpt_configs.num_hidden_layers
        gpt_configs.num_hidden_layers = model_args.num_hidden_layers
        model = GPT2LMHeadModel(gpt_configs)
        
        # for param in model.lm_head.parameters():
        #     param.requires_grad = False

        # load context decoder 
        #### 1. load context ln_f wte wpe weight
        state_dict = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path).state_dict()
        load_model = model.load_state_dict(state_dict, strict=False)
        print("GPT-2 Model Missing Keys: ", load_model.missing_keys)
        print("GPT-2 Model Unexpected Keys: ", load_model.unexpected_keys)

        #### 2. load context decoder layer
        state_dict = {key.replace('transformer.', ''): state_dict[key] for key in state_dict}
        decoder_state = {}
        for name, param in state_dict.items():
            if name.startswith('h'):
                layer_id = int(name.split('.')[1])
                if layer_id >= model_args.feature_extractor_layers:
                    new_layer_id = layer_id - model_args.feature_extractor_layers
                    new_name = str(new_layer_id) +'.' + '.'.join(name.split('.')[2:])
                    decoder_state[new_name] = param
            else:
                print(f"Warning: ignore param {name}!")
        model.transformer.h.load_state_dict(decoder_state, strict=True)

        if model_args.use_vqvae:
            from safetensors.torch import load_file as safe_load
            state_dict = safe_load(model_args.vqvae_model_path)
            model.vqvae_model.load_state_dict(state_dict, strict=True)
            print(f"Already load vqvae model from {model_args.vqvae_model_path} !")
                
            for param in model.vqvae_model.parameters():
                param.requires_grad = False
        else:
            load_feature_extractor = model.feature_extractor.load_state_dict(state_dict, strict=False)
            print("ContextModel feature_extractor Missing Keys: ", load_feature_extractor.missing_keys)
            print("ContextModel feature_extractor Unexpected Keys: ", load_feature_extractor.unexpected_keys)
            for param in model.feature_extractor.parameters():
                param.requires_grad = False

        use_lora = False
        if use_lora:
            ## add lora
            from peft import LoraConfig, get_peft_model
            target_modules = [f'transformer.h.{i}.{j}.c_proj'for i in range(model_args.num_hidden_layers) for j in ['attn', 'mlp']] +\
                  [f'transformer.h.{i}.attn.c_attn'for i in range(model_args.num_hidden_layers)] +\
                      [f'transformer.h.{i}.mlp.c_fc'for i in range(model_args.num_hidden_layers)] +\
                         ['transformer.wpe']
            # 配置 LoRA
            lora_config = LoraConfig(
                r=128,  # LoRA rank
                lora_alpha=256,  # Scaling factor
                target_modules=target_modules,  # GPT-2 关键层的模块名
                lora_dropout=0.0,  # Dropout 概率
                bias="none",  # 是否加偏置
                task_type="CAUSAL_LM"  # GPT-2 是因果语言模型
            )
            model = get_peft_model(model, lora_config)
            for name, param in model.named_parameters():
                if (any(f"transformer.h.{i}." in name for i in range(model_args.num_hidden_layers))) and ('ln_' in name):
                    param.requires_grad = True
                elif ('transformer.ln_f' in name):
                    param.requires_grad = True

        # 计算总体参数
        total_params = sum(p.numel() for p in model.parameters())

        # 计算可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Training new model from scratch - Total Param size={total_params/2**20:.2f}M params")
        logger.info(f"Training new model from scratch - Total Trainable Param size={trainable_params/2**20:.2f}M params")
        logger.info("Total Param is:")
        for name, param in model.named_parameters():
            logger.info(name)

        # 获取并打印可训练参数的名称
        trainable_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        logger.info("Trainable Param is:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)
        if os.path.exists(training_args.output_dir):
            load_checkpoint = os.path.join(training_args.output_dir, 'model.safetensors')
            if os.path.exists(load_checkpoint):
                from safetensors.torch import load_file as safe_load
                state_dict = safe_load(load_checkpoint)
                model.load_state_dict(state_dict, strict=True)
                print(f"Already load checkpoint from {load_checkpoint}!")
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if hasattr(config, "max_position_embeddings"):
        max_pos_embeddings = config.max_position_embeddings
    else:
        # Define a default value if the attribute is missing in the config.
        max_pos_embeddings = 1024

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > max_pos_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, max_pos_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            if max_pos_embeddings > 0:
                block_size = min(1024, max_pos_embeddings)
            else:
                block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        # block_size = min(data_args.block_size, tokenizer.model_max_length)
        block_size = data_args.block_size
    from tqdm import tqdm
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )
    for split, data in lm_datasets.items():
        total_eval_tokens = 0        
        for chunk in data['labels']:
            total_eval_tokens += len([x for x in chunk if x != padding_index])
        logger.info(f'[{split}] Total eval tokens: {total_eval_tokens}')

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets[data_args.eval_subset]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        # data_collator = data_collator,
    )
    if os.environ.get('WANDB_DISABLED') == 'False':
        from utils.wandb_utils import PerplexityCallback
        trainer.add_callback(PerplexityCallback())
        # trainer.add_callback(WandbContextLMCustomCallback())

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager(): 
            
            # quantize_embedding = model.transformer.global_embedding.quantize_features_with_encoder(inputs['input_ids'])
            out = self.compute_loss(model, inputs, return_outputs=True)
            loss = out[1]['loss']
            # contrastive_loss = out[1]['contrastive_loss']
            # cosine_embedding_loss = out[1]['cosine_embedding_loss']
            # mse_loss = out[1]['mse_loss']

            # cosine_similarity = out[1]['cosine_similarity']
            
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            # cosine_similarity = cosine_similarity.mean()
        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss) # Includes normalizing by gas
        # self.state.cosine_similarity = getattr(self.state, "cosine_similarity", [])
        # self.state.contrastive_loss = getattr(self.state, "contrastive_loss", [])
        # self.state.cosine_embedding_loss = getattr(self.state, "cosine_embedding_loss", [])
        # self.state.mse_loss = getattr(self.state, "mse_loss", [])

        # self.state.cosine_similarity.append(cosine_similarity.detach())
        # self.state.contrastive_loss.append(contrastive_loss.detach())
        # self.state.cosine_embedding_loss.append(cosine_embedding_loss.detach())
        # self.state.mse_loss.append(mse_loss.detach())

        # return loss.detach() / self.args.gradient_accumulation_steps / total_nums
        return loss.detach() / self.args.gradient_accumulation_steps 

    from transformers.utils import is_peft_available, SAFE_WEIGHTS_NAME, WEIGHTS_NAME
    from transformers import PreTrainedModel
    from peft import PeftModel

    import safetensors.torch
    TRAINING_ARGS_NAME = "training_args.bin"

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(self.accelerator.unwrap_model(self.model), supported_classes):
                self.accelerator.unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )
            if use_lora:
                self.model.base_model.model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)
            # self.model.save_pretrained(output_dir, safe_serialization=self.args.save_safetensors)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

    trainer.training_step = training_step.__get__(trainer)
    trainer._save = _save.__get__(trainer)

    from typing import Dict
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()

            # logs['cosine_similarity'] = torch.tensor(self.state.cosine_similarity).mean().item()
            # logs['contrastive_loss'] = torch.tensor(self.state.contrastive_loss).mean().item()
            # logs['cosine_embedding_loss'] = torch.tensor(self.state.cosine_embedding_loss).mean().item()
            # logs['mse_loss'] = torch.tensor(self.state.mse_loss).mean().item()

            # if os.environ.get('WANDB_DISABLED') == 'True':
                # del self.state.contrastive_loss
                # del self.state.cosine_embedding_loss
                # del self.state.mse_loss
                # del self.state.cosine_similarity


            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    trainer._maybe_log_save_evaluate = _maybe_log_save_evaluate.__get__(trainer)


    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity


        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()