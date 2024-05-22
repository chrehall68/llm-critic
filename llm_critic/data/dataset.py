import datasets
import llm_critic.data.crawler as crawler
import json
from typing import Dict, List, Union
import pandas as pd
import os


def merge_neurips(prefix: str) -> Dict[str, Union[List[str], List[bool]]]:
    result = {"title": [], "abstract": [], "accepted": []}
    for file_name in os.listdir(prefix):
        data: Dict = json.loads(open(f"{prefix}/{file_name}").read())

        assert "title" in data
        assert "abstract" in data
        assert "accepted" in data
        result["title"].append(data["title"])
        result["abstract"].append(data["abstract"])
        result["accepted"].append(data["accepted"])
    return result


def load_dataset(
    download: bool = False, year_start: int = 2013, year_end: int = 2017
) -> datasets.Dataset:
    """
    Loads the dataset used in the experiments

    Parameters:
        - download: bool - whether to download the dataset or prepare the dataset locally
        - year_start: int - the starting year for neurips papers
        - year_end: int - the ending year, inclusive, for neurips papers
    """

    if download:
        return datasets.load_dataset("chreh/peer_read_neurips")

    # prepare dataset from scratch
    # first, load peer read without neurips
    peer_read_without_neurips = datasets.load_dataset("allenai/peer_read", "reviews")
    peer_read_without_neurips = [
        peer_read_without_neurips[el] for el in peer_read_without_neurips
    ]
    peer_read_without_neurips = datasets.concatenate_datasets(peer_read_without_neurips)
    kept_columns = ["abstract", "accepted", "title"]
    peer_read_without_neurips = peer_read_without_neurips.remove_columns(
        [
            column
            for column in peer_read_without_neurips.features
            if column not in kept_columns
        ]
    )

    # use our crawler to scrape accepted neurips papers
    if not os.path.exists("./tmp"):
        crawler.main("./tmp", year_start, year_end)

    # turn json files into dataframes
    dataframes = []
    for year in os.listdir("./tmp"):
        dataframes.append(pd.DataFrame.from_dict(merge_neurips(f"./tmp/{year}")))
    neurips = pd.concat(dataframes)
    neurips = datasets.Dataset.from_pandas(neurips)
    neurips = neurips.remove_columns(
        [column for column in neurips.features if column not in kept_columns]
    )
    assert (
        neurips.features == peer_read_without_neurips.features
    ), f"Neurips features: {neurips.features}, PeerRead features: {peer_read_without_neurips.features}"

    # finally, merge datasets
    return datasets.concatenate_datasets([peer_read_without_neurips, neurips])
