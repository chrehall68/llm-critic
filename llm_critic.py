import pandas as pd
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import random
from typing import Dict
from tqdm import tqdm
import pickle
from constants import *
from utils import load_dataset


# globals
responses = {}


# workflow functions
def was_correct(decoded: str, entry: Dict[str, int]) -> bool:
    return LABEL_MAP[entry["accepted"]].lower() in decoded.lower()


def to_zero_shot_prompt(entry: Dict[str, str]) -> str:
    prompt = f"""Please determine whether NeurIPS should accept the following paper based on its abstract.\n\nAbstract: {entry['abstractText']}"""
    return prompt


def to_n_shot_prompt(
    n: int,
    entry: Dict[str, str],
    ds,
    entries,
    supports_system: bool,
    tokenizer: AutoTokenizer,
) -> str:
    system = (
        "You are a NeurIPS reviewer with many years of experience reviewing papers. "
        + "You can tell whether a paper will be accepted just by looking at its abstract.\n"
        + 'For example, given "Abstract: This paper is an example rejected abstract", you might respond "Reviewer decision: Reject"\n'
        + 'As another example, given "Abstract: This paper is an example accepted abstract", you might respond "Reviewer decision: Accept"\n'
    )
    examples = ""
    for i in range(n):
        examples += (
            to_zero_shot_prompt(ds.iloc[entries[i]])
            + LABEL_MAP[ds.iloc[entries[i]]["accepted"]]
            + "\n\n"
        )
    prompt = to_zero_shot_prompt(entry)
    if supports_system:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": system + "\n\n" + prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def workflow(idx, ds, model, verbose: bool = False) -> bool:
    prompt = ds.iloc[idx]["prompt"]

    # encode input, move it to cuda, then generate
    encoded_input = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    original_length = encoded_input.shape[-1]

    outputs = model.generate(
        encoded_input,
        **GENERATION_ARGS,
        pad_token_id=tokenizer.eos_token_id,
    )

    # log the prompt and response if verbose
    if verbose:
        print(tokenizer.decode(outputs[0]))

    decoded = tokenizer.decode(outputs[0, original_length:])
    correct = was_correct(decoded, ds.iloc[idx])

    if decoded not in responses:
        responses[decoded] = []
    responses[decoded].append(idx)

    if verbose:
        print(
            "The model was",
            "correct" if correct else "incorrect",
            " - responded",
            tokenizer.decode(outputs[0, original_length:]),
            "and answer should have been",
            LABEL_MAP[ds.iloc[idx]["accepted"]],
        )
    return correct


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, choices=[model for model in MODEL_MAP.keys()])
parser.add_argument("shot", type=int, choices=[0, 1, 5])
parser.add_argument(
    "--quantized",
    type=str,
    default="None",
    required=False,
    choices=["int8", "nf4", "fp4"],
)

if __name__ == "__main__":
    args = parser.parse_args()

    # load and merge dataset
    ds = load_dataset()

    # apply chat template, if necessary
    model_name = MODEL_MAP[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if CHAT_TEMPLATES[args.model] is not None:
        tokenizer.chat_template = CHAT_TEMPLATES[args.model]

    # load model
    config = None
    if args.quantized == "int8":
        config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantized == "fp4":
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="fp4",
        )
    elif args.quantized == "nf4":
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="sequential",
        quantization_config=config,
    )

    # setup
    n_examples = args.shot
    entries = random.choices(list(range(len(ds))), k=n_examples)
    ds["prompt"] = ds["abstractText"].map(
        lambda e: to_n_shot_prompt(
            n_examples,
            {"abstractText": e},
            ds,
            entries,
            supports_system=SYSTEM_SUPPORTED[args.model],
            tokenizer=tokenizer,
        )
    )
    del ds["abstractText"]
    ds["valid"] = [
        tokenizer(prompt, return_tensors="pt").input_ids.shape[1] < MAX_LEN
        for prompt in ds["prompt"]
    ]

    # run experiment
    num_correct = 0
    n_invalid = 0
    for idx in (prog := tqdm(range(len(ds)))):
        if idx in entries:
            continue  # don't include items that were in the examples
        if not ds["valid"].iloc[idx]:
            n_invalid += 1
            continue  # don't include items that are too long due to mistakes in dataset

        correct = workflow(idx, ds, model)
        if correct:
            num_correct += 1
        prog.set_postfix_str(f"acc: {num_correct/(idx+1):.3f}")

    # log results
    pickle.dump(
        responses,
        open(
            f"{model_name[model_name.index('/')+1:]}_{n_examples}responses.pk",
            "wb",
        ),
    )
    with open(f"{n_examples}_shot.txt", "a") as file:
        file.write(f"{model_name} : {num_correct}/{len(ds)-len(entries)-n_invalid}\n")

    # print results up till now
    with open(f"{n_examples}_shot.txt", "r") as file:
        print(file.read())
