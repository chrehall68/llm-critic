import pandas as pd
import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AutoConfig,
)
import accelerate
import torch
import random
from typing import Dict
from tqdm import tqdm
import pickle

# globals
responses = {}
LABEL_MAP = {0: "Reject", 1: "Accept"}


# workflow functions
def was_correct(decoded: str, entry: Dict[str, int]) -> bool:
    return LABEL_MAP[entry["accepted"]].lower() in decoded.lower()


def to_zero_shot_prompt(entry: Dict[str, str]) -> str:
    prompt = f"""For the following abstract, please decide whether to accept or reject the paper. Please respond "Accept" or "Reject" before elaborating:\n{entry['abstractText']}\n\nDecision (Accept/Reject):\n"""
    return prompt


def to_n_shot_prompt(
    n: int, entry: Dict[str, str], ds, entries, with_system: bool = False
) -> str:
    system = "You are a NeurIPS reviewer with many years of experience reviewing papers. You can tell whether a paper will be accepted just by looking at its abstract.\n\n"
    examples = ""
    for i in range(n):
        examples += (
            to_zero_shot_prompt(ds.iloc[entries[i]])
            + LABEL_MAP[ds.iloc[entries[i]]["accepted"]]
            + "\n\n"
        )
    prompt = to_zero_shot_prompt(entry)
    if with_system:
        return system + examples + prompt
    return examples + prompt


def workflow(idx, ds, model, verbose: bool = False) -> bool:
    prompt = ds.iloc[idx]["prompt"]

    # encode input, move it to cuda, then generate
    encoded_input = tokenizer(prompt, return_tensors="pt")
    encoded_input = {item: val.cuda() for item, val in encoded_input.items()}
    generation = model.generate(
        **encoded_input,
        max_new_tokens=1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    # log the prompt and response if verbose
    if verbose:
        print(tokenizer.batch_decode(generation)[0])

    decoded = tokenizer.decode(generation[0, -1])
    correct = was_correct(decoded, ds.iloc[idx])

    if decoded not in responses:
        responses[decoded] = []
    responses[decoded].append(idx)

    if verbose:
        print(
            "The model was",
            "correct" if correct else "incorrect",
            " - responded",
            tokenizer.decode(generation[0, -1]),
            "and answer should have been",
            LABEL_MAP[ds.iloc[idx]["accepted"]],
        )
    return correct


parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, choices=["falcon", "llama", "mistral", "orca"])
parser.add_argument("shot", type=int, choices=[0, 1, 5])
parser.add_argument(
    "--device", type=int, help="cuda device to use", required=False, default=-1
)
parser.add_argument(
    "--full",
    required=False,
    type=bool,
    help="Whether to use full fp16 precision",
    default=False,
)
parser.add_argument(
    "--system",
    required=False,
    type=bool,
    help="Whether to use system prompts",
    default=False,
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.device != -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # load and merge dataset
    df = pd.read_pickle("./parsed_pdf.h5")
    reviews_df = pd.read_pickle("./reviews_pdf.h5")
    merged_df = df.merge(
        reviews_df,
        left_on=["title", "abstractText", "accepted"],
        right_on=["title", "abstractText", "accepted"],
        how="outer",
    )
    final_df = merged_df[merged_df["accepted"].notna()]
    final_df = final_df[final_df["abstractText"].notna()]
    final_df = final_df[final_df["title"].notna()]
    del final_df["name"]
    del final_df["authors"]
    del final_df["creator"]
    del final_df["emails"]
    del final_df["referenceMentions"]
    del final_df["references"]
    final_df["accepted"] = final_df["accepted"].astype(int)

    # load model
    model_map = {
        "falcon": "tiiuae/falcon-7b-instruct",
        "llama": "meta-llama/Llama-2-7b-chat-hf",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "orca": "microsoft/Orca-2-7b",
    }
    model_name = model_map[args.model]

    config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not args.full:
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config,  # device_map="auto"
        )
    else:
        config = AutoConfig.from_pretrained(model_name)
        with accelerate.init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.bfloat16
            )
        model.tie_weights()
        dev_map = accelerate.infer_auto_device_map(
            model,
            max_memory={0: "11GB", 1: "7GB"},
            no_split_module_classes=[
                "LlamaDecoderLayer",
                "MistralDecoderLayer",
                "FalconDecoderLayer",
            ],
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map=dev_map
        )

    # setup
    n_examples = args.shot
    entries = random.choices(list(range(len(final_df))), k=n_examples)
    final_df["prompt"] = final_df["abstractText"].map(
        lambda e: to_n_shot_prompt(
            n_examples, {"abstractText": e}, final_df, entries, args.system
        )
    )

    # run experiment
    num_correct = 0
    for idx in (prog := tqdm(range(len(final_df)))):
        if idx in entries:
            continue  # don't include items that were in the examples

        correct = workflow(idx, final_df, model)
        if correct:
            num_correct += 1
        prog.set_postfix_str(f"acc: {num_correct/(idx+1):.3f}")

    # log results
    tail = "" if not args.full else "_full"
    tail += "" if not args.system else "_system"
    pickle.dump(
        responses,
        open(
            f"{model_name[model_name.index('/')+1:]}_{n_examples}responses{tail}.pk",
            "wb",
        ),
    )
    with open(f"{n_examples}_shot{tail}.txt", "a") as file:
        file.write(f"{model_name} : {num_correct}/{len(final_df)-len(entries)}\n")

    # print results up till now
    with open(f"{n_examples}_shot{tail}.txt", "r") as file:
        print(file.read())

