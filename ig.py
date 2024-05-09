import pandas as pd
import captum.attr as attr
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import random
from argparse import ArgumentParser
import llm_critic
import utils
from constants import *


# set up argument parser
parser = ArgumentParser()
parser.add_argument(
    "model",
    type=str,
    choices=["llama", "gemma", "galactica"],
    help="the model to evaluate",
)
parser.add_argument(
    "shot", type=int, choices=[0, 1, 5], help="How many examples to give"
)
parser.add_argument(
    "--steps",
    type=int,
    default=512,
    help="How many steps to use when approximating integrated gradients",
    required=False,
)
parser.add_argument("samples", type=int, help="How many different samples to take")
parser.add_argument(
    "--batch_size",
    type=int,
    default=2,
    help="Internal batch size to use when calculating integrated gradients",
    required=False,
)
parser.add_argument("--items", type=int, nargs="+", required=False, default=-1)
parser.add_argument(
    "--quantized",
    type=str,
    default="None",
    required=False,
    choices=["int8", "nf4", "fp4"],
)


if __name__ == "__main__":
    args = parser.parse_args()

    # load and merge dataset
    ds = utils.load_dataset()

    # load chat template
    model_name = args.model
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

    # get examples
    entries = random.choices(list(range(len(ds))), k=args.shot)

    if args.items != -1:
        assert len(args.items) == args.samples

    sample = 0
    while sample < args.samples:
        if args.items != -1:
            entry = ds.iloc[args.items[sample]]
        else:
            entry = ds.iloc[random.randint(0, len(ds))]
        # make prompt
        prompt = llm_critic.to_n_shot_prompt(
            args.shot,
            entry,
            ds,
            entries,
            SYSTEM_SUPPORTED[model_name],
            tokenizer=tokenizer,
        )
        print(prompt)

        # get tokens for later
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

        # calculate which label the model responds w/ (so we can calculate ig for that label)
        with torch.no_grad():
            outputs = model(tokens).logits[0, -1]

        if torch.argmax(outputs) == TOKEN_MAP[model_name][ACCEPT]:
            label = ACCEPT
        elif torch.argmax(outputs) == TOKEN_MAP[model_name][REJECT]:
            label = REJECT
        else:
            print("failed!")
            continue  # try again

        # replace the normal pytorch embeddings (which only take ints) to interpretable embeddings
        # (which are compatible with the float inputs that integratedgradients gives)
        interpretable_emb = attr.configure_interpretable_embedding_layer(
            LAYER_MAP[model_name][0](model), LAYER_MAP[model_name][1]
        )

        # calculate inputs and baselines
        input_embs = interpretable_emb.indices_to_embeddings(tokens).cuda()
        baselines = torch.zeros_like(input_embs).cuda()

        # calculate integrated gradients
        ig = attr.IntegratedGradients(
            lambda inps: utils.softmax_results_embeds(inps, model)
        )
        attributions = ig.attribute(
            input_embs,
            baselines=baselines,
            target=TOKEN_MAP[model_name][label],
            n_steps=args.steps,
            internal_batch_size=args.batch_size,
            return_convergence_delta=True,
        )

        # convert attributions to [len(tokens)] shape
        summarized_attributions = utils.summarize_attributions(attributions[0])
        all_tokens = tokens.squeeze(0)
        all_tokens = list(map(tokenizer.decode, all_tokens))

        # remove the interpretable embedding layer so we can get regular predictions
        attr.remove_interpretable_embedding_layer(
            LAYER_MAP[model_name][0](model), interpretable_emb
        )
        with torch.no_grad():
            predictions = utils.softmax_results(tokens, model)

        MARGIN_OF_ERROR = 0.1  # off by no more than 10 percentage points
        if (
            torch.abs(
                (
                    summarized_attributions.sum()
                    - predictions[0, TOKEN_MAP[model_name][label]]
                )
            )
            >= MARGIN_OF_ERROR
        ):
            print("we are off!!")
            print(
                "we should be getting somewhere near",
                predictions[0, TOKEN_MAP[model_name][label]],
            )
            print("instead, we get", summarized_attributions.sum())

        # make and save html
        SCALE = 2 / summarized_attributions.abs().max()
        attr_vis = utils.CustomDataRecord(
            summarized_attributions * SCALE,  # word attributions
            LABEL_MAP[entry["accepted"]],  # true label
            label,  # attr class
            predictions[0, TOKEN_MAP[model_name][label]],  # attr probability
            summarized_attributions.sum(),  # attr score
            all_tokens,  # raw input ids
            attributions[1],  # convergence delta
        )
        html = utils.visualize_text([attr_vis])

        utils.save_results("ig", html, attributions, model_name, sample)

        sample += 1
