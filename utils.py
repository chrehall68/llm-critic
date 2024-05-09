import random
from typing import Any, Union, List
import torch
from dataclasses import dataclass
from IPython.display import HTML
from captum.attr import visualization as viz
import os
import pandas as pd

# seed experiment
random.seed(2024)


# functions
def softmax_results(inputs: torch.Tensor, model: torch.nn.Module):
    result = model(inputs.cuda()).logits
    return torch.nn.functional.softmax(result[:, -1], dim=-1).cuda()


def softmax_results_embeds(embds: torch.Tensor, model: torch.nn.Module):
    result = model(inputs_embeds=embds.cuda()).logits
    return torch.nn.functional.softmax(result[:, -1], dim=-1).cuda()


def summarize_attributions(attributions):
    with torch.no_grad():
        attributions = attributions.sum(dim=-1).squeeze(0)
        return attributions


# visualization utils
@dataclass
class CustomDataRecord:
    word_attributions: Any
    true_class: str
    attr_class: str
    attr_prob: torch.Tensor
    attr_score: Any
    raw_input_ids: Union[Any, List[str]]
    convergence_delta: Any


def _get_color(attr):
    # clip values to [-2,2])
    attr = max(-2, min(2, attr))
    if attr > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def visualize_text(
    datarecords: List[CustomDataRecord], legend: bool = True
) -> "HTML":  # In quotes because this type doesn't exist in standalone mode
    dom = ["<table width: 100%>"]
    if datarecords[0].convergence_delta is not None:
        rows = [
            "<tr><th>True Label</th>"
            "<th>Attribution Label</th>"
            "<th>Convergence Delta</th>"
            "<th>Attribution Score</th>"
            "<th>Word Importance</th>"
        ]
        for datarecord in datarecords:
            rows.append(
                "".join(
                    [
                        "<tr>",
                        viz.format_classname("{0}".format(datarecord.true_class)),
                        viz.format_classname(
                            "{0} ({1:.2f})".format(
                                datarecord.attr_class, datarecord.attr_prob
                            )
                        ),
                        viz.format_classname(
                            "{0:.2f}".format(datarecord.convergence_delta.item())
                        ),
                        viz.format_classname("{0:.2f}".format(datarecord.attr_score)),
                        viz.format_word_importances(
                            datarecord.raw_input_ids, datarecord.word_attributions
                        ),
                        "<tr>",
                    ]
                )
            )
    else:
        rows = [
            "<tr><th>Predicted Label</th>"
            "<th>Attribution Label</th>"
            "<th>Attribution Score</th>"
            "<th>Word Importance</th>"
        ]
        for datarecord in datarecords:
            rows.append(
                "".join(
                    [
                        "<tr>",
                        viz.format_classname("{0}".format(datarecord.true_class)),
                        viz.format_classname(
                            "{0} ({1:.2f})".format(
                                datarecord.attr_class, datarecord.attr_prob
                            )
                        ),
                        viz.format_classname("{0:.2f}".format(datarecord.attr_score)),
                        viz.format_word_importances(
                            datarecord.raw_input_ids, datarecord.word_attributions
                        ),
                        "<tr>",
                    ]
                )
            )

    if legend:
        dom.append(
            '<div style="border-top: 1px solid; margin-top: 5px; \
            padding-top: 5px; display: inline-block">'
        )
        dom.append("<b>Legend: </b>")

        for value, label in zip([-1, 0, 1], ["Negative", "Neutral", "Positive"]):
            dom.append(
                '<span style="display: inline-block; width: 10px; height: 10px; \
                border: 1px solid; background-color: \
                {value}"></span> {label}  '.format(
                    value=_get_color(value), label=label
                )
            )
        dom.append("</div>")

    dom.append("".join(rows))
    dom.append("</table>")
    html = HTML("".join(dom))

    return html


def save_results(
    experiment_name: str,
    html: HTML,
    attributions: torch.Tensor,
    model_name: str,
    sample_num: int,
):
    # make the directories if necessary
    html_dir = f"html/{experiment_name}/{model_name}"
    pt_dir = f"pt/{experiment_name}/{model_name}"
    if not os.path.exists(html_dir):
        os.makedirs(html_dir)
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)

    # write html data
    open(f"{html_dir}/{sample_num}.html", "w").write(html.data)

    # save pts as well
    torch.save(attributions, f"{pt_dir}/{sample_num}.pt")


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
