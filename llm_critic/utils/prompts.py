from typing import Dict, List
from .constants import ACCEPT, DEFAULT_SYSTEM_PROMPT
from transformers import PreTrainedTokenizer
from datasets import Dataset
import chromadb


def to_zero_shot_prompt(entry: Dict[str, str]) -> str:
    prompt = f"""Please determine whether NeurIPS should accept the following paper based on its abstract.\n\nAbstract: {entry['abstract']}"""
    return prompt


def to_example(entry: Dict[str, str]) -> str:
    return f"""Here's an example of a{'n accepted' if entry['accepted'] == ACCEPT else ' rejected'} abstract:\nAbstract: {entry['abstract']}\n\n"""


def to_n_shot_prompt(
    n: int,
    entry: Dict[str, str],
    ds: Dataset,
    entries,
    supports_system: bool,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> str:
    examples = ""
    for i in range(n):
        examples += to_example(ds[entries[i]])

    prompt = examples + to_zero_shot_prompt(entry)
    if supports_system:
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": system_prompt + "\n\n" + prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def batched_to_n_shot_prompt(
    n: int,
    entries: Dict[str, List[str]],
    ds: Dataset,
    collection: chromadb.Collection,
    supports_system: bool,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> List[str]:
    keys = list(entries.keys())
    num_entries = len(entries[keys[0]])
    examples = {"ids": [[] for _ in range(num_entries)]}
    if n > 0:
        examples = collection.query(
            query_texts=[abstract for abstract in entries["abstract"]],
            include=[],
            n_results=n,
        )

    return [
        to_n_shot_prompt(
            n,
            {k: entries[k][i] for k in keys},
            ds,
            [int(el) for el in examples["ids"][i]],
            supports_system,
            tokenizer,
            system_prompt,
        )
        for i in range(num_entries)
    ]
