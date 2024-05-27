"""
Runs perturbation-based explainability methods (SHAP, LIME)
"""

from transformers import PreTrainedTokenizer
import captum.attr as attr
from typing import List
from captum._utils.models.linear_model import SkLearnLasso
import torch
import torch.nn.functional as F
import random
from llm_critic.utils.constants import *
from llm_critic.utils import (
    setup_experiment,
    setup_parser,
    visualize_text,
    save_results,
    CustomDataRecord,
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


def llama_calculate_special_indices(
    tokens: List[int],
    tokenizer: PreTrainedTokenizer,
    mask_last: bool = False,
    mask_not_abstract: bool = False,
):
    """
    Returns a tensor where 1 means that it is a special token that shouldn't be perturbed
    """
    ret = torch.zeros(len(tokens), device="cuda")
    i = 0
    while i < len(tokens):
        if tokens[i] == tokenizer.bos_token_id:
            ret[i] = 1
        elif tokens[i] == tokenizer.eos_token_id:  # for llama, eos id == padding id
            ret[i] = 1
        elif tokenizer.decode(tokens[i]) == "<|start_header_id|>":
            while tokenizer.decode(tokens[i]) != "<|end_header_id|>":
                ret[i] = 1
                i += 1
            ret[i] = 1  # mark end_header_id as special too

        i += 1

    # llama's tokenizer encodes "Reviewer decision:" as 3 tokens
    if mask_last:
        ret[-3:] = 1

    # mask not abstract masks everything except the abstract
    if mask_not_abstract:
        # first, find index of the last "Abstract:"
        colon = tokenizer(":").input_ids[-1]
        abstract = tokenizer("Abstract").input_ids[-1]
        abstract_start = 0
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == colon and tokens[i - 1] == abstract:
                abstract_start = i + 1
                break
        ret[:abstract_start] = 1
    return ret


def gemma_calculate_special_indices(
    tokens: List[int],
    tokenizer: PreTrainedTokenizer,
    mask_last: bool = False,
    mask_not_abstract: bool = False,
):
    """
    Returns a tensor where 1 means that it is a special token that shouldn't be perturbed
    """
    ret = torch.zeros(len(tokens), device="cuda")
    i = 0
    while i < len(tokens):
        if tokens[i] == tokenizer.bos_token_id:
            ret[i] = 1
        elif tokens[i] == tokenizer.eos_token_id:
            ret[i] = 1
        elif tokens[i] == tokenizer.pad_token_id:
            ret[i] = 1
        elif tokens[i] == tokenizer.unk_token_id:
            ret[i] = 1
        elif tokenizer.decode(tokens[i]) == "<start_of_turn>":
            # mark <start_of_turn> as special
            ret[i] = 1

            # and mark the role as special
            i += 1
            ret[i] = 1
        elif tokenizer.decode(tokens[i]) == "<end_of_turn>":
            ret[i] = 1
        i += 1

    if mask_last:
        # gemma's tokenizer converts "Reviewer decision:" into 3 tokens
        ret[-3:] = 1

    if mask_not_abstract:
        # mask everything before the abstract
        colon = tokenizer(":").input_ids[-1]
        abstract = tokenizer("Abstract").input_ids[-1]
        abstract_start = 0
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == colon and tokens[i - 1] == abstract:
                abstract_start = i + 1
                break
        ret[:abstract_start] = 1

    return ret


def galactica_calculate_special_tokens(
    tokens: List[int],
    tokenizer: PreTrainedTokenizer,
    mask_last: bool = False,
    mask_not_abstract: bool = False,
):
    ret = torch.zeros(len(tokens), device="cuda")
    i = 0

    pounds = 1398
    colon = 48
    instruction = 45750
    response = 9604
    abstract = 1022

    while i < len(tokens):
        if tokens[i] == tokenizer.bos_token_id:
            ret[i] = 1
        elif tokens[i] == tokenizer.eos_token_id:
            ret[i] = 1
        elif tokens[i] == tokenizer.unk_token_id:
            ret[i] = 1
        elif tokens[i] == tokenizer.pad_token_id:
            ret[i] = 1
        elif (
            tokens[i] == pounds
            and (tokens[i + 1] == instruction or tokens[i + 1] == response)
            and tokens[i + 2] == colon
        ):
            ret[i] = 1
            ret[i + 1] = 1
            ret[i + 2] = 1
        i += 1

    if mask_last:
        # galactica turns " Reviewer decision:" into 4 tokens
        ret[-4:] = 1

    if mask_not_abstract:
        abstract_start = 0
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == colon and tokens[i - 1] == abstract:
                abstract_start = i + 1
                break
        ret[:abstract_start] = 1

    return ret


# specific functions
def softmax_results(tokens: torch.Tensor, tokenizer: PreTrainedTokenizer, model):
    with torch.no_grad():
        result = model(
            torch.where(tokens != 0, tokens, tokenizer.eos_token_id).cuda(),
            attention_mask=torch.where(tokens != 0, 1, 0).cuda(),
        ).logits
        ret = torch.nn.functional.softmax(result[:, -1], dim=-1).cpu()
        assert not ret.isnan().any()
    return ret


def get_embeds(tokens: torch.Tensor, model_name: str, model, **kwargs):
    with torch.no_grad():
        return LAYER_MAP[model_name][0](model).embed_tokens(tokens.cuda())


# encode text indices into latent representations & calculate cosine similarity
def exp_embedding_cosine_distance(
    original_inp: torch.Tensor,
    perturbed_inp: torch.Tensor,
    perturbed_interpretable_inp: torch.Tensor,
    model_name: str,
    model,
    **kwargs
):
    original_emb = get_embeds(original_inp, model_name, model)
    perturbed_emb = get_embeds(perturbed_inp, model_name, model)
    distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=-1)
    distance[distance.isnan()] = 0
    ret = torch.exp(-1 * (distance**2) / 2).sum()
    assert not ret.isnan().any()
    return ret
    # return perturbed_interpretable_inp.squeeze().count_nonzero().to(torch.float32)


# binary vector where each word is selected independently and uniformly at random
def bernoulli_perturb(text, dont_perturb: torch.Tensor, **kwargs):
    probs = torch.ones_like(text) * 0.5  # everything has a 50% chance to be perturbed
    # always keep the items that shouldn't be perturbed
    # by giving them a 100% chance to be selected to be kept
    probs[0, dont_perturb.nonzero().squeeze()] = 1
    ret = torch.bernoulli(probs).long()
    return ret


# remove absent tokens based on the intepretable representation sample
def interp_to_input(
    interp_sample: torch.Tensor, original_input: torch.Tensor, **kwargs
):
    ret = original_input.clone()
    # The domain of g is {0, 1}^d, i.e. g acts over absence/presence of the interpretable components.
    # 0 means absence, 1 means presence
    ret[interp_sample.bool().logical_not()] = 0
    return ret


class RetryingSKLearnLasso(SkLearnLasso):
    """
    Continues retrying with lower alphas until it finally succeeds
    in highlighting at least 1 word
    """

    def __init__(self, *args, k: int, alpha: int, **kwargs):
        super().__init__(*args, alpha=alpha, **kwargs)
        self.k = k

    def fit(self, *args, **kwargs):
        # store original alpha
        original_alpha = self.construct_kwargs["alpha"]

        # try once
        ret = super().fit(*args, **kwargs)
        while self.representation().count_nonzero() < self.k:
            # keep retrying by decreasing alpha
            self.construct_kwargs["alpha"] *= 0.8
            ret = super().fit(*args, **kwargs)

        # restore original alpha
        self.construct_kwargs["alpha"] = original_alpha

        # return value (not that meaningful)
        return ret


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
