"""
Combines interpretability outputs for a sample into a single .html file
"""

import os
import argparse

# formatting constants
STARTER = """<table width: 100%><div STYLE="border-top: 1px solid; margin-top: 5px;             padding-top: 5px; display: inline-block"><b>Legend: </b><span STYLE="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(0, 75%, 60%)"></span> Negative  <span STYLE="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(0, 75%, 100%)"></span> Neutral  <span STYLE="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(120, 75%, 50%)"></span> Positive  </div><tr><th>Predicted Label</th><th>Experiment Type</th><th>Word Importance</th></tr>"""
END = """</table>"""
STYLE = """STYLE="border-bottom: 1px solid black;" """

# constants for finding the start of a row
START_STR = "<td><mark"
END_STR = "<tr></table>"

# constants for finding the label
LABEL_START_STR = "</td><td><text"
LABEL_END_STR = ")</b></text>"
LABEL_MAP = {"1": "Accept", "0": "Reject"}


def to_row_format(result: str, expname: str, tdmarkstuff: str):
    return f"""<tr><td>{result}</td><td {STYLE}><b>{expname}</b></td>{tdmarkstuff.replace(START_STR, f"<td {STYLE}><mark")}</tr>\n\n"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    idir = args.input_dir
    odir = args.output_dir

    for model in os.listdir(f"{idir}/ig"):  # ig is arbitrary
        for i in range(50):
            result = STARTER
            if not all(
                map(
                    lambda e: os.path.exists(f"{idir}/{e}/{model}/{i}.html"),
                    ["ig", "lime", "shap"],
                )
            ):
                continue  # don't have all files
            for exp in ["ig", "lime", "shap"]:
                html_file = open(f"{idir}/{exp}/{model}/{i}.html").read()
                start_idx = html_file.index(START_STR)
                last_idx = html_file.index(END_STR) - len(END_STR)

                temp = html_file[
                    html_file.index(LABEL_START_STR)
                    + len(LABEL_START_STR) : html_file.index(LABEL_END_STR)
                ]
                temp = (
                    "<b>"
                    + LABEL_MAP[temp[temp.index("<b>") + 3 : temp.index("(")].strip()]
                    + "</b>"
                )

                result += to_row_format(
                    temp, exp.upper(), html_file[start_idx:last_idx]
                )
            result += END
            if not os.path.exists(f"{odir}/combined/{model}/"):
                os.makedirs(f"{odir}/combined/{model}/")
            open(f"{odir}/combined/{model}/{i}.html", "w").write(result)
