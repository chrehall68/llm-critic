"""
Runs the quantitative critic experiment
"""

import argparse
import torch
from llm_critic.core.constants import *
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from llm_critic.core.utils import load_dataset
import random
from llm_critic.core.llm_critic import to_n_shot_prompt, run_experiment
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, choices=[model for model in MODEL_MAP.keys()])
parser.add_argument("shot", type=int, choices=[0, 1, 5])
parser.add_argument(
    "id", type=int, help="the shard (0 <= id < splits) of the dataset to evaluate on"
)
parser.add_argument("split", type=int, help="How many shards to split the dataset into")
parser.add_argument(
    "--batch_size",
    type=int,
    default=5,
    required=False,
    help="how many samples to process in a batch",
)
parser.add_argument(
    "--quantized",
    type=str,
    default="None",
    required=False,
    choices=["int8", "nf4", "fp4"],
    help="the quantization method/dtype to use, defaults to None",
)
parser.add_argument(
    "--dtype",
    type=str,
    choices=["bfloat16", "float16"],
    default="float16",
    help="the compute dtype to use",
)


if __name__ == "__main__":
    args = parser.parse_args()

    # load and merge dataset
    ds = load_dataset()

    # apply chat template, if necessary
    model_name = MODEL_MAP[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if CHAT_TEMPLATES[args.model] is not None:
        tokenizer.chat_template = CHAT_TEMPLATES[args.model]
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
        torch_dtype=(torch.bfloat16 if args.dtype == "bfloat16" else torch.float16),
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
    ds["valid"] = [
        tokenizer(prompt, return_tensors="pt").input_ids.shape[1] < MAX_LEN
        for prompt in ds["prompt"]
    ]

    # calculate split sizes
    split_sizes = [len(ds) // args.split for _ in range(args.split)]
    remainder = len(ds) % args.split
    for i in range(args.split):
        split_sizes[i] += 1 if remainder > 0 else 0
        remainder -= 1

    # convert split sizes into start,end
    split_starts = split_sizes.copy()
    for i in range(1, args.split):
        split_starts[i] += split_starts[i - 1]
    split_starts.insert(0, 0)
    start, end = split_starts[args.id], split_starts[args.id + 1]
    assert start < end and end - start == split_sizes[args.id]

    # run experiment
    results = run_experiment(start, end, entries, args.batch_size, ds, tokenizer, model)

    # log results
    pickle.dump(
        results.responses,
        open(
            f"{model_name[model_name.index('/')+1:]}_{n_examples}_{args.id}responses.pk",
            "wb",
        ),
    )
    with open(f"{n_examples}_shot.txt", "a") as file:
        file.write(
            f"{model_name} {args.id}: {results.n_correct}/{(end-start)-results.used_entries-results.n_invalid}"
            f", n_invalid: {results.n_invalid}, true_ds_len: {len(ds)}\n"
        )

    # print results up till now
    with open(f"{n_examples}_shot.txt", "r") as file:
        print(file.read())
