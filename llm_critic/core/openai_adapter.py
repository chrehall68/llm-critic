from llm_critic.core.llm_critic import was_correct
from llm_critic.core.constants import GENERATION_ARGS, LABEL_MAP
from llm_critic.core.utils import ExperimentResult
from typing import List, Dict
from pandas import DataFrame
from tqdm import tqdm
from openai import OpenAI
import os
import json


class OpenAIAdapterTokenizer:
    def apply_chat_template(self, messages: List[Dict[str, str]], **kwargs) -> str:
        return json.dumps(messages)


def grade_openai(
    idx: int,
    ds: DataFrame,
    client: OpenAI,
    responses: Dict[str, List[int]],
    verbose: bool = False,
    model: str = "gpt-4o",
) -> int:
    prompt: List[Dict[str, str]] = json.loads(ds.iloc[idx]["prompt"])
    completion = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=GENERATION_ARGS["temperature"],
        max_tokens=GENERATION_ARGS["max_new_tokens"] + 2,
        n=1,  # only generate 1 response
    )

    decoded = completion.choices[0].message.content
    correct = was_correct(decoded, ds.iloc[idx])

    if decoded not in responses:
        responses[decoded] = []
    responses[decoded].append(idx)

    if verbose:
        print(
            "The model was",
            "correct" if correct else "incorrect",
            " - responded",
            decoded,
            "and answer should have been",
            LABEL_MAP[ds.iloc[idx]["accepted"]],
        )
    return 1 if correct else 0


def run_experiment_openai(
    start: int,
    end: int,
    examples: List[int],
    ds: DataFrame,
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
    client = OpenAI(
        organization="org-fXZDgasRHznUjRagVlzSgoS6",
        project=os.environ["OPENAI_API_PROJECT"],
    )
    num_correct = 0
    n_invalid = 0
    used_examples = 0
    responses = {}
    for idx in (prog := tqdm(range(start, end))):
        if idx in examples:
            used_examples += 1
            continue  # don't include items that were in the examples
        if not ds["valid"].iloc[idx]:
            n_invalid += 1
            continue  # don't include items that are too long due to mistakes in dataset

        num_correct += grade_openai(idx, ds, client, responses, verbose)
        prog.set_postfix_str(f"acc: {num_correct/(idx-start+1-used_examples):.3f}")

    return ExperimentResult(
        n_correct=num_correct,
        total_samples=(end - start),
        examples_used=used_examples,
        responses=responses,
        n_invalid=n_invalid,
    )
