import torch
from typing import Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def parse_model_name(model_name: str) -> Tuple[str, str, str]:
    # Format: (base model name)(-chat, instruct, etc if necessary)-(param count)
    parts = model_name.split("-")
    assert len(parts) in {2, 3}, f"invalid or underspecified model name: {model_name}"

    base_model_name = parts[0]
    param_count = parts[-1]

    model_type = ""
    if len(parts) == 3:
        model_type = parts[1]
    return base_model_name, model_type, param_count


def load_model_tokenizer(
    model_name: str,
    eval_mode: bool = True,
    device_map: str = "auto",
    attn_implementation: str = "eager",
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    base_model_name, model_type, param_count = parse_model_name(model_name)

    if base_model_name == "llama2":
        # Model sizes: 7b, 13b, 70b
        # Model types: base (unlabeled), chat
        model_path = f"meta-llama/Llama-2-{param_count.lower()}{'-' + model_type.lower() if model_type else ''}-hf"

    elif base_model_name == "llama3":
        # Model sizes: 8B, 70B
        # Model types: base (unlabeled), Instruct
        model_path = f"meta-llama/Meta-Llama-3-{param_count.upper()}{'-' + model_type.capitalize() if model_type else ''}"

    elif base_model_name == "llama3.1":
        # Model sizes: 8B, 70B, 405B
        # Model types: base (unlabeled), Instruct
        model_path = f"meta-llama/Meta-Llama-3.1-{param_count.upper()}{'-' + model_type.capitalize() if model_type else ''}"

    elif base_model_name == "codellama":
        # Model sizes: 7b, 13b, 34b, 70b
        # Model types: base (unlabeled), Instruct, Python
        model_path = f"meta-llama/CodeLlama-{param_count.lower()}{'-' + model_type.capitalize() if model_type else ''}-hf"

    else:
        raise NotImplementedError("Only implemented models: Llama2, Llama3, Llama3.1, CodeLlama")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    if eval_mode:
        model.eval()
        model.requires_grad_(False)

    return model, tokenizer


def get_layers_heads(model_name: str) -> Tuple[int, int]:
    base_model_name, model_type, param_count = parse_model_name(model_name)

    if base_model_name == "llama2":
        # Model sizes: 7b, 13b, 70b
        # Model types: base (unlabeled), chat
        if param_count.lower() == "7b":
            layers, heads = 32, 32
        elif param_count.lower() == "13b":
            layers, heads = 40, 40
        elif param_count.lower() == "70b":
            layers, heads = 80, 64
        else:
            raise NotImplementedError("Only implemented Llama2 models: 7b, 13b, 70b")

    elif base_model_name == "llama3":
        # Model sizes: 8B, 70B
        # Model types: base (unlabeled), Instruct
        if param_count.upper() == "8B":
            layers, heads = 32, 32
        elif param_count.upper() == "70B":
            layers, heads = 80, 64
        else:
            raise NotImplementedError("Only implemented Llama3 models: 8B, 70B")

    elif base_model_name == "llama3.1":
        # Model sizes: 8B, 70B
        # Model types: base (unlabeled), Instruct
        if param_count.upper() == "8B":
            layers, heads = 32, 32
        elif param_count.upper() == "70B":
            layers, heads = 80, 64
        elif param_count.upper() == "405B":
            layers, heads = 126, 128
        else:
            raise NotImplementedError("Only implemented Llama3 models: 8B, 70B")

    elif base_model_name == "codellama":
        # Model sizes: 7b, 13b, 34b, 70b
        # Model types: base (unlabeled), Instruct, Python
        if param_count.lower() == "7b":
            layers, heads = 32, 32
        elif param_count.lower() == "13b":
            layers, heads = 40, 40
        elif param_count.lower() == "34b":
            layers, heads = 48, 64
        elif param_count.lower() == "70b":
            layers, heads = 80, 64
        else:
            raise NotImplementedError(
                "Only implemented CodeLlama models: 7b, 13b, 34b, 70b"
            )

    else:
        raise NotImplementedError("Only implemented models: Llama2, Llama3, Llama3.1, CodeLlama")

    return layers, heads
