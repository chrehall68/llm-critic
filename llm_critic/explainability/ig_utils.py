from typing import Union
import torch


def softmax_results(
    inputs: torch.Tensor,
    model: torch.nn.Module,
    attention_mask: Union[torch.Tensor, None] = None,
):
    result = model(inputs.cuda(), attention_mask=attention_mask).logits
    return torch.nn.functional.softmax(result[:, -1], dim=-1).cuda()


def softmax_results_embeds(
    embds: torch.Tensor,
    model: torch.nn.Module,
    attention_mask: Union[torch.Tensor, None] = None,
):
    result = model(inputs_embeds=embds.cuda(), attention_mask=attention_mask).logits
    return torch.nn.functional.softmax(result[:, -1], dim=-1).cuda()


def summarize_attributions(attributions):
    """
    Sum up the attributions of every token's embedding dimension
    so that attributions reflect importance of tokens instead
    of importance of tokens' embeddings
    """
    with torch.no_grad():
        attributions = attributions.sum(dim=-1).squeeze(0)
        return attributions
