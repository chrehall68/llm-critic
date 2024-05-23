import datasets
import llm_critic.data.crawler as crawler
import json
from typing import Dict, List, Union
import pandas as pd
import os
import os
import json
from typing import List, Dict, Set, Union
import pandas as pd


def get_title(reviews: Dict) -> str:
    review_title = reviews["title"]
    assert review_title is not None
    return review_title


def get_abstract(reviews: Dict, parsed_pdf: Dict) -> str:
    if "abstract" in reviews and reviews["abstract"] is not None:
        return reviews["abstract"]
    if (
        "abstractText" in parsed_pdf["metadata"]
        and parsed_pdf["metadata"]["abstractText"] is not None
    ):
        return parsed_pdf["metadata"]["abstractText"]
    assert False, "Both review and parsed pdf abstract were null"


def get_accepted(
    reviews: Dict, is_acl: bool, title: str, acl_accepted: str
) -> Union[bool, None]:
    if "accepted" in reviews:
        return reviews["accepted"]
    elif is_acl:
        return title in acl_accepted
    return None


def check_accepted_label(prefix: str) -> bool:
    """
    Returns whether all items in the parsed_pdfs directory
    have a matching review that labels it as accepted or rejected
    """

    # compare via number matching
    def compute_numbers(pre: str, tail: str) -> Set[str]:
        # first, calculate all numbers in the parsed pdfs
        numbers = set()
        items = 0
        for file_name in os.listdir(pre):
            assert tail in file_name, file_name
            num = file_name.strip(tail)
            numbers.add(num)
            items += 1
        assert len(numbers) == items  # make sure all titles were unique
        return numbers

    parsed_pdfs = compute_numbers(prefix + "/parsed_pdfs", ".pdf.json")
    reviews = compute_numbers(prefix + "/reviews", ".json")
    dif = parsed_pdfs.symmetric_difference(reviews)
    return len(dif) == 0


def check_unique_titles(prefix: str) -> bool:
    """
    Returns whether all itmes in the parsed_pdfs and reviews directories
    have a unique string title
    """
    titles = set()
    title_dict = {}
    items = 0
    for file_name in os.listdir(prefix + "/parsed_pdfs"):
        file_id = file_name.strip(".pdf.json")

        parsed_pdf = json.loads(open(f"{prefix}/parsed_pdfs/{file_name}").read())
        reviews = json.loads(open(f"{prefix}/reviews/{file_id}.json").read())

        title = get_title(reviews)
        if title in titles:
            print("title", title, "exists at", title_dict[title], "and", file_name)
        titles.add(title)
        title_dict[title] = file_name
        items += 1
    return len(titles) == items


def run_checks(debug: bool = False):
    all_valid = True
    for dir in os.listdir("data"):
        if os.path.isdir(f"data/{dir}") and dir != "nips_2013-2017":
            for split in ["test", "dev", "train"]:
                prefix = f"data/{dir}/{split}/"
                if not check_accepted_label(prefix):
                    if debug:
                        print(f"no matching review for data/{dir}/{split}")
                    all_valid = False
    if all_valid:
        print("All parsed_pdfs have matching reviews!")

    all_valid = True
    for dir in os.listdir("data"):
        if os.path.isdir(f"data/{dir}") and dir != "nips_2013-2017":
            for split in ["test", "dev", "train"]:
                prefix = f"data/{dir}/{split}/"
                if not check_unique_titles(prefix):
                    if debug:
                        print(f"no unique title for data/{dir}/{split}")
                    all_valid = False
    if all_valid:
        print(
            "Not all parsed_pdfs have unique titles! Make sure not to merge based on titles!"
        )


def merge_pdfs_and_reviews(prefix: str) -> Dict[str, Union[List[str], List[bool]]]:
    result = {"title": [], "abstract": [], "accepted": []}
    # have to use a special case for acl since their reviews don't say accept/reject
    is_acl = "acl_2017" in prefix
    acl_text = open(prefix + "/../../acl_accepted.txt").read()

    # parse
    for file_name in os.listdir(prefix + "/parsed_pdfs"):
        file_id = file_name.strip(".pdf.json")

        # load files
        parsed_pdf: Dict = json.loads(open(f"{prefix}/parsed_pdfs/{file_name}").read())
        reviews: Dict = json.loads(open(f"{prefix}/reviews/{file_id}.json").read())

        # add data
        result["title"].append(get_title(reviews))
        result["accepted"].append(
            get_accepted(reviews, is_acl, get_title(reviews), acl_text)
        )
        result["abstract"].append(get_abstract(reviews, parsed_pdf))
    return result


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
    download: bool = True, year_start: int = 2013, year_end: int = 2019
) -> datasets.Dataset:
    """
    Loads the dataset used in the experiments

    Parameters:
        - download: bool - whether to download the dataset or prepare the dataset locally
        - year_start: int - the starting year for neurips papers
        - year_end: int - the ending year, inclusive, for neurips papers
    """

    if download:
        return datasets.load_dataset("chreh/peer_read_neurips")["train"]

    # prepare dataset from scratch
    # first, load peer read without neurips
    # need to clone their github repository
    if not os.path.exists("./PeerRead"):
        status = os.system("git clone https://github.com/allenai/PeerRead")
        if status != 0:
            raise Exception("Failed to clone Peer Read repository")

    # now, merge all files into a single dict containing
    # - abstract
    # - title
    # - accepted
    kept_columns = ["abstract", "title", "accepted"]
    items = []
    prefix = "PeerRead/data"
    for dir in os.listdir(prefix):
        # when cloning the repository, there is nothing in the
        # nips_2013-2017 directory except a readme
        # so ignore it
        if os.path.isdir(f"{prefix}/{dir}") and dir != "nips_2013-2017":
            for split in ["train", "test", "dev"]:
                items.append(
                    pd.DataFrame.from_dict(
                        merge_pdfs_and_reviews(f"{prefix}/{dir}/{split}")
                    )
                )

    without_neurips = pd.concat(items)
    without_neurips = without_neurips.dropna()
    without_neurips = datasets.Dataset.from_pandas(without_neurips)
    without_neurips = without_neurips.remove_columns(
        [column for column in without_neurips.features if column not in kept_columns]
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
        neurips.features == without_neurips.features
    ), f"Neurips features: {neurips.features}, PeerRead features: {without_neurips.features}"

    # finally, merge datasets
    return datasets.concatenate_datasets([without_neurips, neurips])
