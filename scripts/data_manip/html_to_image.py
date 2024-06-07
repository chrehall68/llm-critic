"""
This script converts html files to images for insertion into papers.

Example usage:

python3 scripts/data_manip/html_to_image.py html/ images/
"""

import html2image
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    for model in os.listdir(f"{args.input_dir}"):
        hti = html2image.Html2Image(
            size=(1080, 1200), output_path=f"{args.output_dir}/{model}/"
        )
        for run in os.listdir(f"{args.input_dir}/{model}"):
            name = run[: run.index(".html")]
            hti.screenshot(
                html_file=f"{args.input_dir}/{model}/{name}.html",
                css_str="",
                save_as=f"{name}.png",
            )
