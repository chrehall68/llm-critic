"""
This file calculates the accept/reject distribution of responses, printing them to stdout.
It also formats that data in a latex table that gets printed to stdout.

Example usage:

python3 scripts/insights/distribution.py processed_results/
"""

import pickle as pk
import os
import argparse


def fn(d):
    accepting = set(filter(lambda a: "accept" in a.lower(), d.keys()))
    rejecting = set(filter(lambda a: "reject" in a.lower(), d.keys()))
    both = set(
        filter(lambda a: "accept" in a.lower() and "reject" in a.lower(), d.keys())
    )
    print(
        f"Unique responses: accepting {len(accepting)}, rejecting {len(rejecting)}, both {len(both)}, total unique responses {len(d.keys())}"
    )
    total_accept = sum(len(d[key]) for key in accepting)
    total_reject = sum(len(d[key]) for key in rejecting)
    total_neither = sum(
        len(d[key]) for key in d if key not in accepting and key not in rejecting
    )
    total = total_accept + total_reject + total_neither
    print(
        f"total accept {total_accept}, total reject {total_reject}, total neither {total_neither}, total {total}"
    )
    print(
        f"accept {total_accept/total*100:.2f}%, reject {total_reject/total*100:.2f}%, neither {total_neither/total*100:.2f}%"
    )
    return {
        "accept": total_accept / total,
        "reject": total_reject / total,
        "neither": total_neither / total,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    args = parser.parse_args()
    data_dir = args.data_dir
    names = ["llama", "galactica", "gemma"]
    models = {model: {} for model in names}
    for file in os.listdir(data_dir):
        if not file.endswith(".pk"):
            continue
        responses = pk.loads(open(data_dir + file, "rb").read())
        print(file)
        shot = file[file.rindex(".") - 1]
        if "llama" in file.lower():
            models["llama"][shot] = fn(responses)
        elif "galactica" in file.lower():
            models["galactica"][shot] = fn(responses)
        elif "gemma" in file.lower():
            models["gemma"][shot] = fn(responses)
        print()

    # tabularize it
    s = "& & \\textbf{Llama 3 8b} & \\textbf{Gemma 7b} & \\textbf{Galactica 6.7b} \\\\\n\\midrule\n"
    for shot in ["0", "1", "5"]:
        for t in ["accept", "reject", "neither"]:
            row = ""
            if t == "accept":
                row = f"{shot} shot"
            row += f" & {t}"
            for model in names:
                row += f" & {models[model][shot][t]*100:.2f}\\%"
            row += " \\\\\n"
            s += row
        if shot != "5":
            s += "\\midrule\n"
    print(s)
