"""
This file calculates metrics for the critic experiment. It outputs a latex table to stdout.

Example usage:

python3 scripts/insights/metrics.py processed_results/
"""

import argparse
import pickle as pk
import os
from sklearn.metrics import (
    matthews_corrcoef,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from llm_critic.data import load_dataset
from llm_critic.utils import ACCEPT, REJECT, MODEL_MAP
from typing import Dict, List, Literal, Union


def calculate_metrics(responses: Dict[str, List[int]]) -> Dict[
    Union[
        Literal["precision"],
        Literal["recall"],
        Literal["f1"],
        Literal["mcc"],
        Literal["kappa"],
    ],
    Union[Dict[Union[Literal["accept"], Literal["reject"]], float], float],
]:
    ds = load_dataset()
    y_true = ds["accepted"]
    y_pred = [-float("inf") for _ in range(len(y_true))]
    for response in responses:
        if "accept" in response.lower():
            label = ACCEPT
        elif "reject" in response.lower():
            label = REJECT
        else:
            label = -1

        for idx in responses[response]:
            y_pred[idx] = label

    # remove the items that were used as examples
    while -float("inf") in y_pred:
        idx = y_pred.index(-float("inf"))
        y_pred.pop(idx)
        y_true.pop(idx)
    assert -float("inf") not in y_pred

    # get rid of the "neither" entries as well
    while -1 in y_pred:
        idx = y_pred.index(-1)
        y_pred.pop(idx)
        y_true.pop(idx)

    # now calculate precision, recall, f1 score
    # it's f1 score since fscore beta defaults to 1 in sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)

    metrics = {"precision": precision, "recall": recall, "f1": f1}
    ret = {metric: {} for metric in metrics}
    for metric in metrics:
        ret[metric]["accept"] = metrics[metric][ACCEPT]
        ret[metric]["reject"] = metrics[metric][REJECT]

    # calculate other metrics too
    ret["kappa"] = cohen_kappa_score(y_true, y_pred)
    ret["mcc"] = matthews_corrcoef(y_true, y_pred)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()
    model_to_shot = {model: {} for model in MODEL_MAP}

    # calculate metrics
    for file in os.listdir(args.input_dir):
        if file.endswith(".pk"):
            responses = pk.loads(open(args.input_dir + "/" + file, "rb").read())
            shot = file[file.rindex(".") - 1]
            model_name = ""
            for model in model_to_shot:
                if model in file.lower():
                    model_name = model
            result = calculate_metrics(responses)

            model_to_shot[model_name][shot] = result

    # construct table
    order = ["precision", "recall", "f1", "kappa", "mcc"]
    tab = "\\begin{tabularx}{\\textwidth}{C|C|C|CCC|CC}\n\\toprule\n"
    tab += "\\textbf{Shot} & \\textbf{Model} & \\textbf{Label} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Kappa} & \\textbf{MCC} \\\\\n\\midrule\n"
    for shot in ["0", "1", "5"]:
        tab += f"{shot} shot"
        for model in model_to_shot:
            r1 = f" & {model} & accept"
            r2 = " & & reject"

            for metric in order:
                if metric in {"precision", "recall", "f1"}:
                    r1 += f" & {model_to_shot[model][shot][metric]['accept']:.3f}"
                    r2 += f" & {model_to_shot[model][shot][metric]['reject']:.3f}"
                else:
                    r1 += f" & {model_to_shot[model][shot][metric]:.3f}"
                    r2 += " & "
            r1 += "\\\\\n"
            r2 += "\\\\\n"
            tab += r1
            tab += r2
        tab += "\\midrule\n"
    tab = tab[: tab.rindex("\\midrule\n")]
    tab += "\\bottomrule\n\\end{tabularx}"
    print(tab)
