import pandas as pd
from dataclasses import dataclass
from llm_critic.utils.constants import SYSTEM_SUPPORTED, MAX_LEN
from pandas import DataFrame
from llm_critic.utils.prompts import to_n_shot_prompt
from transformers import AutoTokenizer
from typing import Dict, List, Tuple


def load_dataset() -> pd.DataFrame:
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
    return final_df


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
    ds: DataFrame,
    n_examples: int,
    examples: List[int],
    model_name: str,
    tokenizer: AutoTokenizer,
    calculate_valid=lambda tok, prompt: tok(
        prompt, return_tensors="pt"
    ).input_ids.shape[1]
    < MAX_LEN,
) -> None:
    ds["prompt"] = ds["abstractText"].map(
        lambda e: to_n_shot_prompt(
            n_examples,
            {"abstractText": e},
            ds,
            examples,
            supports_system=SYSTEM_SUPPORTED[model_name],
            tokenizer=tokenizer,
        )
    )
    ds["valid"] = [calculate_valid(tokenizer, prompt) for prompt in ds["prompt"]]
