import argparse
import os
from llm_critic.utils import MODEL_MAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str)
    args = parser.parse_args()

    # calculate regular split
    s = "& \\textbf{Llama 3 8b} & \\textbf{Gemma 7b} & \\textbf{Galactica 6.7b} \\\\\n\\midrule\n"
    for file in os.listdir(args.directory):
        if os.path.isfile(args.directory + "/" + file) and file.endswith(".txt"):
            txt = open(args.directory + "/" + file).read()
            txt = txt.strip().split("\n")
            # extract the "a/b" part
            nums = list(map(lambda row: row[row.index(" ") + 1 : row.index(",")], txt))
            # turn "a/b" into [int(a), int(b)]
            nums = list(map(lambda row: list(map(int, row.split("/"))), nums))
            # calculate a/b numerically
            nums = list(map(lambda row: row[0] / row[1], nums))

            s += file[: file.index(".txt")].replace("_", "-")
            for model in ["llama", "gemma", "galactica"]:
                for idx, row in enumerate(txt):
                    if MODEL_MAP[model] in row:
                        s += f" & {nums[idx]*100:.2f}\\%"
            s += " \\\\\n"
    print(s)
