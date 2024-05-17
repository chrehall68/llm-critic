import torch


# functions
def softmax_results(inputs: torch.Tensor, model: torch.nn.Module):
    result = model(inputs.cuda()).logits
    return torch.nn.functional.softmax(result[:, -1], dim=-1).cuda()


def softmax_results_embeds(embds: torch.Tensor, model: torch.nn.Module):
    result = model(inputs_embeds=embds.cuda()).logits
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
