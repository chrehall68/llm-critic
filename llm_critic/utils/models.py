from typing import Union, Literal, List, Dict
import torch
from .constants import MODEL_MAP, CHAT_TEMPLATES
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import json


class OpenAIAdapterTokenizer:
    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return json.dumps(messages)


def load_tokenizer(model_name: str):
    """
    Loads and prepares the tokenizer for the model with the given `model_name`

    Arguments:
        - model_name: str - the simplified  model name
    """
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_MAP[model_name], padding_side="left"
    )
    if CHAT_TEMPLATES[model_name] is not None:
        tokenizer.chat_template = CHAT_TEMPLATES[model_name]
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_model(
    model_name: str,
    quantized: Union[
        Literal["None"], Literal["int8"], Literal["fp4"], Literal["nf4"]
    ] = "None",
    dtype: Union[Literal["bfloat16"], Literal["float16"]] = "bfloat16",
):
    """
    Loads and prepares the model, potentially quantizing / cloning the model on different
    GPUs

    Arguments:
        - model_name: str - the simplified model name
        - quantized: str - a string specifying the quantization dtype to use, or None
        - dtype: str - the compute dtype to use, either bfloat16 or float16
    """
    config = None
    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16
    if quantized == "int8":
        config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantized == "fp4":
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="fp4",
        )
    elif quantized == "nf4":
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_name],
        torch_dtype=torch_dtype,
        device_map="sequential",
        quantization_config=config,
    )
    return model
