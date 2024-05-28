from typing import List
import torch
import torch.nn.functional as F
from captum._utils.models.linear_model import SkLearnLasso
from transformers import PreTrainedTokenizer
from llm_critic.utils import LAYER_MAP

MASKED_FLAG = -1


def llama_calculate_special_indices(
    tokens: List[int],
    tokenizer: PreTrainedTokenizer,
    mask_last: bool = False,
    mask_not_abstract: bool = False,
):
    """
    Returns a tensor where 1 means that it the token at that position shouldn't be perturbed
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
            torch.where(tokens != MASKED_FLAG, tokens, tokenizer.eos_token_id).cuda(),
            attention_mask=torch.where(tokens != MASKED_FLAG, 1, 0).cuda(),
        ).logits
        ret = torch.nn.functional.softmax(result[:, -1], dim=-1).cpu()
        assert not ret.isnan().any()
    return ret


def get_embeds(tokens: torch.Tensor, model_name: str, model, **kwargs):
    with torch.no_grad():
        # prevent index errors by replacing -1 (masked flag)
        # with an actual index
        embeds: torch.Tensor = LAYER_MAP[model_name][0](model).embed_tokens(
            tokens.where(tokens != MASKED_FLAG, 0).cuda()
        )
        # but we don't want to use embed[0] since that could have meaning
        # so we set those instead to 0
        embeds[tokens == MASKED_FLAG] = 0
        return embeds


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
    ret[interp_sample.bool().logical_not()] = MASKED_FLAG
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
