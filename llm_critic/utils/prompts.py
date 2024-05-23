from typing import Dict
from llm_critic.utils.constants import ACCEPT
from transformers import AutoTokenizer
from datasets import Dataset


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
        examples += to_example(ds[entries[i]])

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
