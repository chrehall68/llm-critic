"""
Runs the quantitative critic experiment
"""

import argparse
import torch
from llm_critic.core.constants import *
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from llm_critic.core.utils import load_dataset
import random
from llm_critic.core.llm_critic import *

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
    num_correct = 0
    n_invalid = 0
    used_entries = 0
    cur_lst = []
    responses = {}
    for idx in (prog := tqdm(range(start, end))):
        if idx in entries:
            used_entries += 1
            continue  # don't include items that were in the examples
        if not ds["valid"].iloc[idx]:
            n_invalid += 1
            continue  # don't include items that are too long due to mistakes in dataset

        cur_lst.append(idx)
        if len(cur_lst) >= args.batch_size:  # only compute when batch is full
            num_correct += grade(cur_lst, ds, tokenizer, model, responses)
            cur_lst.clear()
        prog.set_postfix_str(f"acc: {num_correct/(idx-start+1-used_entries):.3f}")
    if len(cur_lst) > 0:  # handle any leftovers
        num_correct += grade(cur_lst, ds, tokenizer, model, responses)
        cur_lst.clear()

    # log results
    pickle.dump(
        responses,
        open(
            f"{model_name[model_name.index('/')+1:]}_{n_examples}_{args.id}responses.pk",
            "wb",
        ),
    )
    with open(f"{n_examples}_shot.txt", "a") as file:
        file.write(
            f"{model_name} {args.id}: {num_correct}/{(end-start)-used_entries-n_invalid}, n_invalid: {n_invalid}, true_ds_len: {len(ds)}\n"
        )

    # print results up till now
    with open(f"{n_examples}_shot.txt", "r") as file:
        print(file.read())
