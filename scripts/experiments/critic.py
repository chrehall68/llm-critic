"""
Runs the quantitative critic experiment

Example usage:

python3 scripts/experiments/critic.py --model llama --shot 0 --id 0 --splits 1 --batch_size 5 --dtype float16
"""

from llm_critic.core.llm_critic import run_experiment
from llm_critic.utils import setup_parser, setup_experiment, MODEL_MAP
import pickle

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
    tokenizer, model, ds, entries, start, end = setup_experiment(args)

    # run experiment
    results = run_experiment(start, end, entries, args.batch_size, ds, tokenizer, model)

    # log results
    pickle.dump(
        results.responses,
        open(
            f"{model_name[model_name.index('/')+1:]}_{args.shot}_{args.id}responses.pk",
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
