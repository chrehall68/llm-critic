"""
This script combines sharded pickled results into a single pickle file

Example usage:

python3 scripts/data_manip/combine_quantitative.py
"""

from collections import defaultdict
import os
import re
import pickle as pk
from typing import Dict, List


# combine outputs into a single file
def combine_metrics(
    file_name: str, data_dir: str = "./results", output_dir: str = "./processed"
):
    sharded = open(f"{data_dir}/{file_name}").read().strip()
    lines = sharded.split("\n")
    model_valids = defaultdict(int)
    model_totals = defaultdict(int)
    model_invalid = defaultdict(int)
    for line in lines:
        items = line.split(" ")
        name, result, n_invalid = items[0], items[2], items[4]
        result = [int(el) for el in result.strip(",").split("/")]
        n_invalid = int(n_invalid.strip(","))

        model_totals[name] += result[0]
        model_valids[name] += result[1]
        model_invalid[name] += n_invalid

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/{file_name}", "w") as file:
        for model in model_valids:
            file.write(
                f"{model} {model_totals[model]}/{model_valids[model]}, n_invalid: {model_invalid[model]}\n"
            )


# combine pickles into single dictionaries and serialize them
def combine_pickles(
    shot: int, data_dir: str = "./results", output_dir: str = "./processed"
):
    files = defaultdict(list)
    for file in os.listdir(data_dir):
        matches = re.findall(f"(.*)_{shot}_[0-9]+", file)
        if len(matches) > 0:
            # this is a match, store model: file
            files[matches[0]].append(file)

    for model in files:
        m = defaultdict(list)
        dicts: List[Dict[str, List[int]]] = [
            pk.loads(open(f"{data_dir}/{file}", "rb").read()) for file in files[model]
        ]
        for d in dicts:
            for key, val in d.items():
                m[key].extend(val)
        pk.dump(m, open(f"{output_dir}/{model}_{shot}.pk", "wb"))


if __name__ == "__main__":
    for shot in [0, 1, 5]:
        combine_metrics(f"{shot}_shot.txt")
        combine_pickles(shot)
