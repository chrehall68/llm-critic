"""
Runs the quantitative critic experiment,
but with openai-compatible models

Example usage:

python3 scripts/experiments/openai_critic.py --model meta-llama/Meta-Llama-3-8B-Instruct --shot 0 --id 0 --splits 1 --batch_size 5
"""

from llm_critic.core.openai_adapter import run_experiment_openai
from llm_critic.utils import setup_parser, setup_experiment_openai
import pickle
from llm_critic.utils.constants import MODEL_MAP
import asyncio

parser = setup_parser()
parser.add_argument(
    "--batch_size",
    type=int,
    default=5,
    required=False,
    help="how many samples to process in a batch",
)


if __name__ == "__main__":
    args = parser.parse_args()
    model_name = MODEL_MAP[args.model]

    # setup experiment
    tokenizer, ds, entries, start, end = setup_experiment_openai(args)

    # run experiment
    results = asyncio.run(
        run_experiment_openai(model_name, start, end, entries, args.batch_size, ds)
    )

    # log results
    pickle.dump(
        results.responses,
        open(
            f"{model_name.replace('/', '_')}_{args.shot}_{args.id}responses.pk",
            "wb",
        ),
    )
    with open(f"{args.shot}_shot.txt", "a") as file:
        file.write(
            f"{model_name} {args.id}: {results.n_correct}/{(end-start)-results.examples_used-results.n_invalid}"
            f", n_invalid: {results.n_invalid}, true_ds_len: {len(ds)}\n"
        )

    # print results up till now
    with open(f"{args.shot}_shot.txt", "r") as file:
        print(file.read())
