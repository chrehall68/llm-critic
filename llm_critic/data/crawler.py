#!/usr/bin/python

import json
import argparse
import os
import re
import urllib3
from tqdm import tqdm


def main(base_path: str, year_start: int, year_end: int):
    """ "
    Crawl Neurips articles from [year_start, year_end]
    and output JSON files for each article in the directory `{base_path}/{year}`

    JSON files will contain the following fields:
        - abstract
        - accepted
        - authors
        - title
        - url
    """
    base_url = "https://papers.neurips.cc/paper_files/paper/"

    for year in range(year_start, year_end + 1):
        # make the output directory
        output_dir = base_path + "/" + str(year)
        os.makedirs(output_dir, exist_ok=True)

        # get the base url for that year
        year_url = base_url + str(year)
        year_html = urllib3.request("GET", year_url).data.decode()
        papers = re.findall('<a title="paper title" href="(.*)">', year_html)

        for paper in tqdm(papers, desc=f"Year {year}"):
            # id that will be used to name the output file
            paper_id = re.findall("/hash/(.*)-Abstract.html", paper)

            # get and parse html
            paper_url = "https://papers.neurips.cc" + paper
            paper_html = urllib3.request("GET", paper_url).data.decode()
            title = re.findall("<h4>(.*)</h4>\\s*<p>\\s*Part of", paper_html)
            authors = re.findall("<h4>Authors</h4>\\s*<p>\\s*<i>(.*)</i>", paper_html)
            abstract = re.findall(
                "<h4>Abstract</h4>(?:.+)<p>(.*)</p>\\s*[</p>]*\\s*</div>\\s*</div>",
                paper_html,
                re.S,
            )

            # checks
            assert len(paper_id) == 1, paper_html
            assert len(title) == 1, paper_html
            assert len(authors) == 1, paper_html
            assert len(abstract) == 1, paper_html
            paper_id = paper_id[0]
            title = title[0]
            authors = authors[0]
            abstract = abstract[0].strip().strip("</p>")

            # write data
            json.dump(
                {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "url": paper_url,
                    "accepted": True,
                },
                open(output_dir + "/" + paper_id + ".json", "w"),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--year_start", type=int, required=False, default=2013)
    parser.add_argument("--year_end", type=int, required=False, default=2017)
    args = parser.parse_args()
    main(args.output_dir, args.year_start, args.year_end)
