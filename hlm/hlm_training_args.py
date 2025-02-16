from typing import Optional
from transformers.training_args import TrainingArguments
from dataclasses import field


class HLMTrainingArguments(TrainingArguments):
    frozen_layers: str = field(
        default=None,
        metadata={  "help": "Freezing the first or last layers", 
                    "choices": ["first", "last", None]}
    )
    input_layers: int = field(
        default=3, metadata={"help": "Input layer numbers"}
    )

    ctx_layers: int = field(
        default=6, metadata={"help": "ctx prediction layer numbers"}
    )
    
        # VAE_configs

    codebook_size: Optional[int] = field(
        default=False,
    )

    training_type: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "training_type"
            ),
            "choices": ['codebook', 'full', 'after_input_layer_include_cb', 'except_codebook', 'ours', 'only_output_layer', 'after_input_layer_exclude_cb', 'only_ctx_layer_include_cb', 'only_ctx_layer_exclude_cb'],
        },
    )

    vq_type: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "vq_type"
            ),
            "choices": ['VectorQuantize', 'SimVQ', 'LFQ', 'ResidualVQ', 'GroupedResidualVQ', 'RandomProjectionQuantizer', 'ResidualSimVQ'],
        },
    )

    embedding_dim: Optional[int] = field(
        default=False,
    )

    commitment_cost: Optional[float] = field(
        default=False,
    )

    tokens_per_group: Optional[int] = field(
        default=False,
    )
