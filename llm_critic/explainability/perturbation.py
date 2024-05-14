import argparse
import captum.attr as attr
from captum._utils.models.linear_model import SkLearnLasso
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch.nn.functional as F
import random
import llm_critic.core.utils as utils
from llm_critic.core.constants import *
import llm_critic.core.llm_critic as llm_critic

parser = argparse.ArgumentParser()
parser.add_argument(
    "model",
    type=str,
    help="the model to use",
    choices=[
        "llama",
        "gemma",
        "galactica",
    ],
)
parser.add_argument(
    "experiment_type",
    type=str,
    help="the experiment type to run",
    choices=["lime", "shap"],
)
parser.add_argument(
    "n_examples", type=int, help="the number of examples to use", choices=[0, 1, 5]
)
parser.add_argument(
    "n_perturbation_samples",
    type=int,
    help="the number of samples to use in perturbation methods",
)
parser.add_argument(
    "n_samples", type=int, help="the number of times to run the experiment"
)
parser.add_argument("--items", type=int, nargs="+", required=False, default=-1)
parser.add_argument(
    "--quantized",
    type=str,
    default="None",
    required=False,
    choices=["int8", "nf4", "fp4"],
)


# specific functions
def softmax_results(tokens: torch.Tensor):
    with torch.no_grad():
        if tokenizer.bos_token_id is not None:
            tokens[0, 0] = tokenizer.bos_token_id
        result = model(
            torch.where(tokens != 0, tokens, tokenizer.eos_token_id).cuda(),
            attention_mask=torch.where(tokens != 0, 1, 0).cuda(),
        ).logits
        ret = torch.nn.functional.softmax(result[:, -1], dim=-1).cpu()
        assert not ret.isnan().any()
    return ret


def get_embeds(tokens: torch.Tensor):
    with torch.no_grad():
        return LAYER_MAP[model_name][0](model).embed_tokens(tokens.cuda())


# encode text indices into latent representations & calculate cosine similarity
def exp_embedding_cosine_distance(original_inp, perturbed_inp, _, **kwargs):
    original_emb = get_embeds(original_inp)
    perturbed_emb = get_embeds(perturbed_inp)
    distance = 1 - F.cosine_similarity(original_emb, perturbed_emb, dim=-1)
    distance[distance.isnan()] = 0
    ret = torch.exp(-1 * (distance**2) / 2).sum()
    assert not ret.isnan().any()
    return ret


# binary vector where each word is selected independently and uniformly at random
def bernoulli_perturb(text, **kwargs):
    probs = torch.ones_like(text) * 0.5
    probs[0, 0] = 0  # don't get rid of the start token
    ret = torch.bernoulli(probs).long()
    return ret


# remove absent tokens based on the intepretable representation sample
def interp_to_input(interp_sample, original_input, **kwargs):
    ret = original_input.clone()
    ret[interp_sample.bool()] = 0
    return ret


if __name__ == "__main__":
    # parse args
    args = parser.parse_args()
    model_name = args.model
    experiment_type = args.experiment_type.upper()
    n_examples = args.n_examples
    n_perturbation_samples = args.n_perturbation_samples
    n_samples = args.n_samples

    # load dataset
    ds = utils.load_dataset()

    # apply chat template
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[model_name])
    if CHAT_TEMPLATES[model_name] is not None:
        tokenizer.chat_template = CHAT_TEMPLATES[model_name]

    # load model
    config = None
    if args.quantized == "int8":
        config = BitsAndBytesConfig(load_in_8bit=True)
    elif args.quantized == "fp4":
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="fp4",
        )
    elif args.quantized == "nf4":
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[model_name],
        torch_dtype=torch.bfloat16,
        device_map="sequential",
        quantization_config=config,
    )

    # calculate entries
    entries = random.choices(list(range(len(ds))), k=n_examples)

    if args.items != -1:
        assert len(args.items) == n_samples
    for sample in range(n_samples):
        if args.items != -1:
            entry = ds.iloc[args.items[sample]]
        else:
            entry = ds.iloc[random.randint(0, len(ds))]
        # make prompt
        prompt = llm_critic.to_n_shot_prompt(
            n_examples,
            entry,
            ds,
            entries,
            SYSTEM_SUPPORTED[model_name],
            tokenizer=tokenizer,
        )
        print(prompt)

        # get tokens for later
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

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
                softmax_results,
                interpretable_model=SkLearnLasso(
                    alpha=3e-4, selection="cyclic", max_iter=5000
                ),
                similarity_func=exp_embedding_cosine_distance,
                perturb_func=bernoulli_perturb,
                perturb_interpretable_space=True,
                from_interp_rep_transform=interp_to_input,
                to_interp_rep_transform=None,
            )
        elif experiment_type == SHAP:
            attributer = attr.KernelShap(softmax_results)
        else:
            raise Exception("Invalid Experiment Type")
        attributions = attributer.attribute(
            tokens,
            target=TOKEN_MAP[model_name][label],
            n_samples=n_perturbation_samples,
            show_progress=True,
        )

        # verify attributions
        assert attributions.nonzero().numel() != 0
        all_tokens = tokens.squeeze(0)
        all_tokens = list(map(tokenizer.decode, all_tokens))

        # get predictions for comparison
        with torch.no_grad():
            predictions = softmax_results(tokens)

        # save attributions
        SCALE = 2 / attributions.abs().max()
        attr_vis = utils.CustomDataRecord(
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
        html = utils.visualize_text([attr_vis])

        utils.save_results(
            experiment_type.lower(),
            html,
            attributions,
            model_name,
            sample,
        )
