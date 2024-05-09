import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import random
from typing import Dict, List
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


def workflow(idxs: List[int], ds, model, verbose: bool = False) -> int:
    prompts = list(ds.iloc[idxs]["prompt"])

    # encode input, move it to cuda, then generate
    encoded_input = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    original_length = encoded_input.input_ids.shape[-1]

    outputs = model.generate(
        encoded_input.input_ids,
        attention_mask=encoded_input.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        **GENERATION_ARGS,
    )

    n_correct = 0
    for item_num, idx in enumerate(idxs):
        # log the prompt and response if verbose
        if verbose:
            print(tokenizer.decode(outputs[item_num]))

        decoded = tokenizer.decode(outputs[item_num, original_length:])
        correct = was_correct(decoded, ds.iloc[idx])

        if decoded not in responses:
            responses[decoded] = []
        responses[decoded].append(idx)
        n_correct += 1

        if verbose:
            print(
                "The model was",
                "correct" if correct else "incorrect",
                " - responded",
                tokenizer.decode(outputs[item_num, original_length:]),
                "and answer should have been",
                LABEL_MAP[ds.iloc[idx]["accepted"]],
            )
    return n_correct


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
    type="str",
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
    for idx in (prog := tqdm(range(start, end))):
        if idx in entries:
            used_entries += 1
            continue  # don't include items that were in the examples
        if not ds["valid"].iloc[idx]:
            n_invalid += 1
            continue  # don't include items that are too long due to mistakes in dataset

        cur_lst.append(idx)
        if len(cur_lst) >= args.batch_size:  # only compute when batch is full
            num_correct += workflow(cur_lst, ds, model)
            cur_lst.clear()
        prog.set_postfix_str(f"acc: {num_correct/(idx-start+1):.3f}")
    if len(cur_lst) > 0:  # handle any leftovers
        num_correct += workflow(cur_lst, ds, model)
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
            f"{model_name} {args.id}: {num_correct}/{len(ds)-used_entries-n_invalid}\n"
        )

    # print results up till now
    with open(f"{n_examples}_shot.txt", "r") as file:
        print(file.read())
