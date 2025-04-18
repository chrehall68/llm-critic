from dataclasses import dataclass
from .constants import SYSTEM_SUPPORTED, MAX_LEN, DEFAULT_SYSTEM_PROMPT
from datasets import Dataset
from .prompts import to_n_shot_prompt
from transformers import PreTrainedTokenizer
from typing import Dict, List, Tuple, Optional


@dataclass
class ExperimentResult:
    n_correct: int
    total_samples: int
    examples_used: int
    responses: Dict[str, List[int]]
    n_invalid: int


def split(n: int, m: int, k: int) -> Tuple[int, int]:
    """
    Splits n items into m groups of roughly n/m
    and returns the k'th group's start/end for

    Arguments:
        - n: int - the number of items to split
        - m: int - the number of groups to split into
        - k: int - the group (0 <= k < m) to return the
            start/end items for

    Returns:
        - Tuple[int, int]: - a tuple (start, end) where
            all items in [start, end) should be in the k'th group
    """

    # calculate split sizes
    split_sizes = [n // m for _ in range(m)]
    remainder = n % m
    for i in range(m):
        split_sizes[i] += 1 if remainder > 0 else 0
        remainder -= 1

    # convert split sizes into start,end
    split_starts = split_sizes.copy()
    for i in range(1, m):
        split_starts[i] += split_starts[i - 1]
    # put a 0 at the start since the first group will need to start at 0
    split_starts.insert(0, 0)
    start, end = split_starts[k], split_starts[k + 1]
    assert start < end and end - start == split_sizes[k]
    return (start, end)


def preprocess_dataset(
    ds: Dataset,
    n_examples: int,
    examples: List[int],
    model_name: str,
    tokenizer: PreTrainedTokenizer,
    system_prompt: Optional[str] = None,
    calculate_valid=lambda tok, prompt: tok(
        prompt, return_tensors="pt"
    ).input_ids.shape[1]
    < MAX_LEN,
) -> Dataset:
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT

    ds = ds.map(
        lambda e: {
            "prompt": to_n_shot_prompt(
                n_examples,
                e,
                ds,
                examples,
                supports_system=SYSTEM_SUPPORTED[model_name],
                tokenizer=tokenizer,
                system_prompt=system_prompt,
            )
        },
        load_from_cache_file=False,
    )
    ds = ds.map(
        lambda e: {"valid": calculate_valid(tokenizer, e["prompt"])},
        load_from_cache_file=False,
    )
    ds = ds.map(
        lambda e: {"accepted": 1 if e["accepted"] else 0}, load_from_cache_file=False
    )
    return ds
