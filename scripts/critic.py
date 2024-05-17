"""
Runs the quantitative critic experiment
"""

from llm_critic.core.llm_critic import run_experiment
from llm_critic.utils import setup_parser, setup_experiment
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

    # setup experiment
    model_name, tokenizer, model, ds, entries, start, end = setup_experiment(args)

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
            f"{model_name} {args.id}: {results.n_correct}/{(end-start)-results.used_entries-results.n_invalid}"
            f", n_invalid: {results.n_invalid}, true_ds_len: {len(ds)}\n"
        )

    # print results up till now
    with open(f"{args.shot}_shot.txt", "r") as file:
        print(file.read())
