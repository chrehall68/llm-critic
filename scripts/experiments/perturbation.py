"""
Runs perturbation-based explainability methods (SHAP, LIME)
"""

import captum.attr as attr
import torch
import random
from llm_critic.utils.constants import *
from llm_critic.utils import (
    setup_experiment,
    setup_parser,
    visualize_text,
    save_results,
    CustomDataRecord,
)
from llm_critic.explainability.lime_utils import (
    softmax_results,
    RetryingSKLearnLasso,
    exp_embedding_cosine_distance,
    bernoulli_perturb,
    llama_calculate_special_indices,
    galactica_calculate_special_tokens,
    gemma_calculate_special_indices,
    interp_to_input,
)

parser = setup_parser()
parser.add_argument(
    "--experiment_type",
    required=True,
    type=str,
    help="the experiment type to run",
    choices=["lime", "shap"],
)
parser.add_argument(
    "--n_perturbation_samples",
    required=True,
    type=int,
    help="the number of samples to use in perturbation methods",
)
parser.add_argument(
    "--n_samples",
    required=True,
    type=int,
    help="the number of times to run the experiment",
)
parser.add_argument(
    "--batch_size",
    type=int,
    required=False,
    default=1,
    help="how many samples to process in a batch",
)
parser.add_argument(
    "--items",
    type=int,
    nargs="+",
    required=False,
    default=-1,
    help="The specific samples to evaluate on, or -1 if not provided",
)


if __name__ == "__main__":
    # parse args
    args = parser.parse_args()
    model_name = args.model
    experiment_type = args.experiment_type.upper()
    n_perturbation_samples = args.n_perturbation_samples
    n_samples = args.n_samples

    # experiment setup
    tokenizer, model, ds, entries, start, end = setup_experiment(args, n_samples)

    # sample setup
    if args.items != -1:
        assert len(args.items) == n_samples
        samples = args.items
    else:
        samples = random.choices(
            list(set(list(range(len(ds)))) - set(entries)), k=n_samples
        )

    for i in range(start, end):
        entry = ds[samples[i]]

        # get tokens for later
        # add special tokens = False to prevent adding an extra BOS token
        tokens = tokenizer(
            entry["prompt"], return_tensors="pt", add_special_tokens=False
        ).input_ids.to("cuda")

        # calculate which label the model responds w/ (so we can perturb for that label)
        with torch.no_grad():
            outputs = model(tokens).logits[0, -1]

        if torch.argmax(outputs) == TOKEN_MAP[model_name][ACCEPT]:
            label = ACCEPT
        elif torch.argmax(outputs) == TOKEN_MAP[model_name][REJECT]:
            label = REJECT
        else:
            print("failed!")
            continue  # try again

        # run experiment
        LIME = "LIME"
        SHAP = "SHAP"
        if experiment_type == LIME:
            attributer = attr.LimeBase(
                lambda inp: softmax_results(inp, tokenizer, model),
                interpretable_model=RetryingSKLearnLasso(
                    k=5, alpha=0.01, selection="cyclic", max_iter=5000
                ),
                similarity_func=exp_embedding_cosine_distance,
                perturb_func=bernoulli_perturb,
                perturb_interpretable_space=True,
                from_interp_rep_transform=interp_to_input,
                to_interp_rep_transform=None,
            )

            # calculate items that shouldn't be perturbed
            list_tokens = tokenizer(entry["prompt"], add_special_tokens=False).input_ids
            fn_map = {
                "llama": llama_calculate_special_indices,
                "gemma": gemma_calculate_special_indices,
                "galactica": galactica_calculate_special_tokens,
            }
            dont_perturb = fn_map[model_name](
                list_tokens, tokenizer, mask_last=True, mask_not_abstract=True
            )

            # now attribute
            attributions = attributer.attribute(
                tokens,
                target=TOKEN_MAP[model_name][label],
                n_samples=n_perturbation_samples,
                show_progress=True,
                perturbations_per_eval=args.batch_size,
                # kwargs
                model_name=model_name,
                model=model,
                dont_perturb=dont_perturb,
            )
        elif experiment_type == SHAP:
            attributer = attr.KernelShap(
                lambda inp: softmax_results(inp, tokenizer, model)
            )
            attributions = attributer.attribute(
                tokens,
                target=TOKEN_MAP[model_name][label],
                n_samples=n_perturbation_samples,
                show_progress=True,
                perturbations_per_eval=args.batch_size,
            )
        else:
            raise Exception("Invalid Experiment Type")

        # verify attributions
        assert attributions.nonzero().numel() != 0
        all_tokens = tokens.squeeze(0)
        all_tokens = list(map(tokenizer.decode, all_tokens))

        # get predictions for comparison
        with torch.no_grad():
            predictions = softmax_results(tokens, tokenizer, model)

        # save attributions
        SCALE = 2 / attributions.abs().max()
        attr_vis = CustomDataRecord(
            attributions[0] * SCALE,  # word attributions
            LABEL_MAP[entry["accepted"]],  # true class
            label,  # attr class
            predictions[0, TOKEN_MAP[model_name][label]],  # attr probability
            attributions.sum(),  # attr score
            all_tokens,  # raw input ids
            (
                abs(predictions[0, TOKEN_MAP[model_name][label]] - attributions.sum())
                if experiment_type == SHAP
                else None
            ),
        )
        html = visualize_text([attr_vis])

        save_results(
            experiment_type.lower(),
            html,
            attributions,
            model_name,
            i,
        )
