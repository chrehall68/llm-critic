"""
Combines interpretability outputs for a sample into a single .html file
"""

import os

# formatting constants
STARTER = """<table width: 100%><div STYLE="border-top: 1px solid; margin-top: 5px;             padding-top: 5px; display: inline-block"><b>Legend: </b><span STYLE="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(0, 75%, 60%)"></span> Negative  <span STYLE="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(0, 75%, 100%)"></span> Neutral  <span STYLE="display: inline-block; width: 10px; height: 10px;                 border: 1px solid; background-color:                 hsl(120, 75%, 50%)"></span> Positive  </div><tr><th>Predicted Label</th><th>Experiment Type</th><th>Word Importance</th></tr>"""
END = """</table>"""
STYLE = """STYLE="border-bottom: 1px solid black;" """

# constants for finding the start of a row
START_STR = "<td><mark"
END_STR = "<tr></table>"

# constants for finding the label
LABEL_START_STR = "<tr><td><text"
LABEL_END_STR = "</b></text>"


def to_row_format(result: str, expname: str, tdmarkstuff: str):
    return f"""<tr><td>{result}</td><td {STYLE}><b>{expname}</b></td>{tdmarkstuff.replace(START_STR, f"<td {STYLE}><mark")}</tr>\n\n"""


if __name__ == "__main__":
    prefix = "./html"  # input/output dir

    for model in os.listdir(f"{prefix}/ig"):  # ig is arbitrary
        for i in range(50):
            result = STARTER
            if not all(
                map(
                    lambda e: os.path.exists(f"{prefix}/{e}/{model}/{i}.html"),
                    ["ig", "lime", "shap"],
                )
            ):
                continue  # don't have all files
            for exp in ["ig", "lime", "shap"]:
                html_file = open(f"{prefix}/{exp}/{model}/{i}.html").read()
                start_idx = html_file.index(START_STR)
                last_idx = html_file.index(END_STR) - len(END_STR)

                temp = ""
                if exp == "ig":
                    temp = html_file[
                        html_file.index(LABEL_START_STR)
                        + 8 : html_file.index(LABEL_END_STR)
                    ]
                result += to_row_format(
                    temp, exp.upper(), html_file[start_idx:last_idx]
                )
            result += END
            if not os.path.exists(f"{prefix}/combined/{model}/"):
                os.makedirs(f"{prefix}/combined/{model}/")
            open(f"{prefix}/combined/{model}/{i}.html", "w").write(result)