from llm_critic.utils.experiments import load_dataset, preprocess_dataset, split
from llm_critic.utils.models import load_model, load_tokenizer
import llm_critic.utils.seed  # needed for the random seed
from argparse import ArgumentParser, Namespace
from llm_critic.utils.constants import MODEL_MAP
import random


def setup_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        choices=[model for model in MODEL_MAP.keys()],
    )
    parser.add_argument("--shot", required=True, type=int, choices=[0, 1, 5])
    parser.add_argument(
        "--id",
        required=False,
        type=int,
        default=0,
        help="the shard (0 <= id < splits) of the dataset to evaluate on",
    )
    parser.add_argument(
        "--splits",
        required=False,
        type=int,
        default=1,
        help="How many shards to split the dataset into",
    )
    parser.add_argument(
        "--quantized",
        type=str,
        default="None",
        required=False,
        choices=["int8", "nf4", "fp4"],
        help="the quantization method/dtype to use, defaults to None",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["bfloat16", "float16"],
        default="float16",
        help="the compute dtype to use",
    )
    return parser


def setup_experiment(args: Namespace, n: int = -1):
    """
    Sets up the experiment by loading the model/tokenizer and
    preprocessing the dataset. Returns a tuple of
    `(model_name, tokenizer, model, ds, entries, start, end)`
    """
    # apply chat template, if necessary
    tokenizer = load_tokenizer(args.model)
    model = load_model(args.model, quantized=args.quantized, dtype=args.dtype)

    # load and preprocess dataset
    ds = load_dataset()
    n_examples = args.shot
    entries = random.choices(list(range(len(ds))), k=n_examples)
    preprocess_dataset(ds, n_examples, entries, args.model, tokenizer)
    if n == -1:
        start, end = split(len(ds), args.splits, args.id)
    else:
        start, end = split(n, args.splits, args.id)

    return tokenizer, model, ds, entries, start, end
