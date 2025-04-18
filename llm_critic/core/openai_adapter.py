from .llm_critic import was_correct
from llm_critic.utils import ExperimentResult, GENERATION_ARGS, LABEL_MAP
from typing import List, Dict
from datasets import Dataset
from tqdm import tqdm
from openai import AsyncOpenAI
import os
import json
import asyncio


async def grade_openai(
    idx: int,
    ds: Dataset,
    client: AsyncOpenAI,
    responses: Dict[str, List[int]],
    verbose: bool = False,
    model: str = "gpt-4o",
) -> int:
    prompt: List[Dict[str, str]] = json.loads(ds[idx]["prompt"])
    completion = await client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=GENERATION_ARGS["temperature"],
        max_tokens=GENERATION_ARGS["max_new_tokens"] + 2,
        # max_tokens=10000,  # TODO: make this dynamically enabled for reasoning models
        n=1,  # only generate 1 response
    )

    decoded = completion.choices[0].message.content
    if not decoded:
        print("No response from model")
        print("But reasoning was", completion.choices[0].message.reasoning_content)
        return 0
    correct = was_correct(decoded, ds[idx])

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
            LABEL_MAP[ds[idx]["accepted"]],
        )
    return 1 if correct else 0


async def run_experiment_openai(
    model: str,
    start: int,
    end: int,
    examples: List[int],
    batch_size: int,
    ds: Dataset,
    verbose: bool = False,
) -> ExperimentResult:
    """
    Run the experiment, deprecated in favor of creating/submitting a batch and grading the results

    Arguments:
        - start: int - the index of the first sample to evaluate on, inclusive
        - end: int - the index of the last sample to evaluate on, exclusive
        - examples: List[int] - a list of indices that are reserved for examples
        - ds: Dataset - the dataset
        - verbose: bool = False - whether or not to print outputs when running the experiment
    """
    # run experiment
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        project=os.environ.get("OPENAI_PROJECT"),
        base_url=os.environ["OPENAI_BASE_URL"],
    )
    num_correct = 0
    n_invalid = 0
    used_examples = 0
    responses = {}
    tasks = []
    sema = asyncio.Semaphore(batch_size)

    async def inner(f):
        async with sema:
            return await f

    for idx in tqdm(range(start, end)):
        if idx in examples:
            used_examples += 1
            continue  # don't include items that were in the examples
        if not ds["valid"][idx]:
            n_invalid += 1
            continue  # don't include items that are too long due to mistakes in dataset

        tasks.append(inner(grade_openai(idx, ds, client, responses, verbose, model)))

    # wait for all tasks to complete
    prog = tqdm(range(len(tasks)))
    total = 0
    for task, _ in zip(asyncio.as_completed(tasks), prog):
        num_correct += await task
        total += 1
        prog.set_postfix_str(f"acc: {num_correct/total:.2f}")

    return ExperimentResult(
        n_correct=num_correct,
        total_samples=(end - start),
        examples_used=used_examples,
        responses=responses,
        n_invalid=n_invalid,
    )


def create_batch(
    ds: Dataset, examples: List[int], out_file: str, model: str = "gpt-4o"
) -> None:
    with open(out_file, "w") as file:
        for i in range(len(ds)):
            if i in examples:
                continue  # don't bother with items that were in the examples

            # process into a batch
            body = {
                "custom_id": f"task-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    # This is what you would have in your Chat Completions API call
                    "model": model,
                    "temperature": GENERATION_ARGS["temperature"],
                    "messages": json.loads(ds[i]["prompt"]),
                    # +2 for "Reviewer decision"
                    "max_tokens": GENERATION_ARGS["max_new_tokens"] + 2,
                },
            }
            file.write(json.dumps(body))
            file.write("\n")
