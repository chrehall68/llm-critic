"""
Runs Integrated Gradients

Example usage:

python3 scripts/experiments/ig.py --model llama --shot 0 --id 0 --splits 1 \\
    --batch_size 2 --dtype float16 --samples 10 --steps 512
"""

import captum.attr as attr
import torch
from tqdm import tqdm
import random
from llm_critic.utils import (
    setup_experiment,
    setup_parser,
    TOKEN_MAP,
    LAYER_MAP,
    ACCEPT,
    REJECT,
    LABEL_MAP,
    CustomDataRecord,
    visualize_text,
    save_results,
)
from llm_critic.explainability.ig_utils import (
    softmax_results,
    softmax_results_embeds,
    summarize_attributions,
)


# set up argument parser
parser = setup_parser()
parser.add_argument(
    "--samples", required=True, type=int, help="How many different samples to take"
)
parser.add_argument(
    "--steps",
    type=int,
    default=512,
    help="How many steps to use when approximating integrated gradients",
    required=False,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    help="Internal batch size to use when calculating integrated gradients",
    required=False,
)
parser.add_argument("--items", type=int, nargs="+", required=False, default=-1)

if __name__ == "__main__":
    args = parser.parse_args()

    # experiment setup
    tokenizer, model, ds, entries, start, end = setup_experiment(args, args.samples)

    # sample setup
    if args.items != -1:
        assert len(args.items) == args.samples
        samples = args.items
    else:
        samples = random.choices(
            list(set(list(range(len(ds)))) - set(entries)), k=args.samples
        )

    # run integrated gradients on the samples
    for i in tqdm(range(start, end)):
        entry = ds[samples[i]]

        # get tokens for later
        # add special tokens = False to prevent adding an extra BOS token
        tokens = tokenizer(
            entry["prompt"], return_tensors="pt", add_special_tokens=False
        ).input_ids.to("cuda")

        # calculate which label the model responds w/ (so we can calculate ig for that label)
        with torch.no_grad():
            outputs = model(tokens).logits[0, -1]

        if torch.argmax(outputs) == TOKEN_MAP[args.model][ACCEPT]:
            label = ACCEPT
        elif torch.argmax(outputs) == TOKEN_MAP[args.model][REJECT]:
            label = REJECT
        else:
            print("failed!")
            continue  # neither is the model's output

        # replace the normal pytorch embeddings (which only take ints) to interpretable embeddings
        # (which are compatible with the float inputs that integratedgradients gives)
        true_model = LAYER_MAP[args.model][0](model)
        interpretable_emb = attr.configure_interpretable_embedding_layer(
            true_model, LAYER_MAP[args.model][1]
        )

        # calculate inputs and baselines
        input_embs = interpretable_emb.indices_to_embeddings(tokens).cuda()
        baselines = torch.zeros_like(input_embs).cuda()

        # calculate integrated gradients
        ig = attr.IntegratedGradients(lambda inps: softmax_results_embeds(inps, model))
        attributions = ig.attribute(
            input_embs,
            baselines=baselines,
            target=TOKEN_MAP[args.model][label],
            n_steps=args.steps,
            internal_batch_size=args.batch_size,
            return_convergence_delta=True,
        )

        # convert attributions to [len(tokens)] shape
        summarized_attributions = summarize_attributions(attributions[0])
        all_tokens = tokens.squeeze(0)
        all_tokens = list(map(tokenizer.decode, all_tokens))

        # remove the interpretable embedding layer so we can get regular predictions
        attr.remove_interpretable_embedding_layer(true_model, interpretable_emb)
        with torch.no_grad():
            predictions = softmax_results(tokens, model)

        MARGIN_OF_ERROR = 0.1  # off by no more than 10 percentage points
        if (
            torch.abs(
                (
                    summarized_attributions.sum()
                    - predictions[0, TOKEN_MAP[args.model][label]]
                )
            )
            >= MARGIN_OF_ERROR
        ):
            print("we are off!!")
            print(
                "we should be getting somewhere near",
                predictions[0, TOKEN_MAP[args.model][label]],
            )
            print("instead, we get", summarized_attributions.sum())

        # make and save html
        SCALE = 2 / summarized_attributions.abs().max()
        attr_vis = CustomDataRecord(
            summarized_attributions * SCALE,  # word attributions
            LABEL_MAP[entry["accepted"]],  # true label
            label,  # attr class
            predictions[0, TOKEN_MAP[args.model][label]],  # attr probability
            summarized_attributions.sum(),  # attr score
            all_tokens,  # raw input ids
            attributions[1],  # convergence delta
        )
        html = visualize_text([attr_vis])

        save_results("ig", html, attributions, args.model, i)
