from transformers import AutoTokenizer
from typing import Dict, List
from llm_critic.utils.constants import *
from llm_critic.utils import load_dataset, ExperimentResult
from tqdm import tqdm
from datasets import Dataset


def was_correct(decoded: str, entry: Dict[str, int]) -> bool:
    return LABEL_MAP[entry["accepted"]].lower() in decoded.lower()


def grade(
    idxs: List[int],
    ds: Dataset,
    tokenizer: AutoTokenizer,
    model,
    responses: Dict[str, List[int]],
    verbose: bool = False,
) -> int:
    prompts = list(ds[idxs]["prompt"])

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
        correct = was_correct(decoded, ds[idx])

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
                LABEL_MAP[ds[idx]["accepted"]],
            )
    return n_correct


def run_experiment(
    start: int,
    end: int,
    examples: List[int],
    batch_size: int,
    ds: Dataset,
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
        if not ds[idx]["valid"]:
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
