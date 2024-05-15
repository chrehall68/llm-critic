from transformers import AutoTokenizer
from typing import Dict, List
from llm_critic.core.constants import *
from llm_critic.core.utils import load_dataset, ExperimentResult
from tqdm import tqdm
from pandas import DataFrame


# workflow functions
def was_correct(decoded: str, entry: Dict[str, int]) -> bool:
    return LABEL_MAP[entry["accepted"]].lower() in decoded.lower()


def to_zero_shot_prompt(entry: Dict[str, str]) -> str:
    prompt = f"""Please determine whether NeurIPS should accept the following paper based on its abstract.\n\nAbstract: {entry['abstractText']}"""
    return prompt


def to_example(entry: Dict[str, str]) -> str:
    return f"""Here's an example of a{'n accepted' if entry['accepted'] == ACCEPT else ' rejected'} abstract:\nAbstract: {entry['abstractText']}\n\n"""


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
        examples += to_example(ds.iloc[entries[i]])

    prompt = examples + to_zero_shot_prompt(entry)
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


def grade(
    idxs: List[int], ds, tokenizer, model, responses: dict, verbose: bool = False
) -> int:
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
        if correct:
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


def run_experiment(
    start: int,
    end: int,
    examples: List[int],
    batch_size: int,
    ds: DataFrame,
    tokenizer: AutoTokenizer,
    model,
    verbose: bool = False,
) -> ExperimentResult:
    """
    Run the experiment

    Arguments:
        - start: int - the index of the first sample to evaluate on, inclusive
        - end: int - the index of the last sample to evaluate on, exclusive
        - examples: List[int] - a list of indices that are reserved for examples
        - batch_size: int - the batch size to use when evaluating
        - ds: DataFrame - the dataset
        - tokenizer: AutoTokenizer - the tokenizer to use
        - model - the LLM to use
        - verbose: bool = False - whether or not to print outputs when running the experiment
    """
    # run experiment
    num_correct = 0
    n_invalid = 0
    used_examples = 0
    cur_lst = []
    responses = {}
    for idx in (prog := tqdm(range(start, end))):
        if idx in examples:
            used_examples += 1
            continue  # don't include items that were in the examples
        if not ds["valid"].iloc[idx]:
            n_invalid += 1
            continue  # don't include items that are too long due to mistakes in dataset

        cur_lst.append(idx)
        if len(cur_lst) >= batch_size:  # only compute when batch is full
            num_correct += grade(cur_lst, ds, tokenizer, model, responses, verbose)
            cur_lst.clear()
        prog.set_postfix_str(f"acc: {num_correct/(idx-start+1-used_examples):.3f}")
    if len(cur_lst) > 0:  # handle any leftovers
        num_correct += grade(cur_lst, ds, tokenizer, model, responses, verbose)
        cur_lst.clear()

    return ExperimentResult(
        n_correct=num_correct,
        total_samples=(end - start),
        examples_used=used_examples,
        responses=responses,
        n_invalid=n_invalid,
    )
