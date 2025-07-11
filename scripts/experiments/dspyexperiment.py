import llm_critic.core.dspy_adapter as adapter
import dspy
from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
from dspy.predict.chain_of_thought import ChainOfThought
from dspy.predict.predict import Predict
import argparse
import os
import pickle as pk


class PeerReadSignature(dspy.Signature):
    abstract = dspy.InputField(desc="The paper's abstract")
    status = dspy.OutputField(
        desc='The paper\'s status, either "accepted" or "rejected"'
    )


class CoTModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = ChainOfThought(PeerReadSignature)

    def forward(self, abstract: str):
        return self.prog(abstract=abstract)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ZeroShotModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = Predict(PeerReadSignature)

    def forward(self, **kwargs):
        return self.prog(**kwargs)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--n", type=int, default=-1)

    # compatible with local llms too
    parser.add_argument(
        "--model", type=str, help="The model to run, without the leading openai/"
    )
    args = parser.parse_args()

    # configure LM
    client = dspy.LM(
        model="openai/" + args.model,
        max_tokens=512,
        temperature=0.0,
        api_key=os.environ["OPENAI_API_KEY"],
        api_base=os.environ.get("OPENAI_BASE_URL"),
        model_type="chat",
        seed=2024,
    )
    dspy.settings.configure(lm=client)

    # load ds
    ds = adapter.PeerReadDspy(trainsize=0.1, devsize=0.05, inputs=["abstract"])
    test_ds = ds.test[: args.n] if args.n != -1 else ds.test
    print(
        f"Train set size: {len(ds.train)}, dev set size: {len(ds.dev)}, test set size: {len(ds.test)}"
    )

    # get all tokens and count them
    # now sum up lengths
    print("Total tokens in test dataset ~", sum(len(e["abstract"]) for e in ds.test))

    prompter = MIPROv2(
        metric=adapter.peerread_metric,
        prompt_model=client,
        task_model=client,
        num_threads=args.concurrency,
    )
    module = ZeroShotModule() if not args.cot else CoTModule()
    optimized_program = prompter.compile(
        student=module,
        trainset=ds.train,
        num_trials=10,
        max_bootstrapped_demos=1,
        max_labeled_demos=5,
    )

    optimized_program.save("output_program")
    module = optimized_program

    # run experiment
    try:
        score, outputs, results = adapter.run_experiment(
            module=module, ds=test_ds, n_threads=args.concurrency
        )
    except:
        print(client.inspect_history())

    pk.dump((score, outputs, results), open("results.pk", "wb"))
