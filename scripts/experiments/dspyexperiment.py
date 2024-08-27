from tokenizers import Tokenizer
import llm_critic.core.dspy_adapter as adapter
import dspy
import dspy.teleprompt
from dspy.predict.asynchronous.acot import AChainOfThought
from dspy.predict.asynchronous.apredict import APredict
import argparse
import os
import asyncio

VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "*")
VLLM_API_BASE = os.environ.get("VLLM_API_BASE", "http://localhost:8001/v1/")


class PeerReadSignature(dspy.Signature):
    abstract = dspy.InputField(desc="The paper's abstract")
    status = dspy.OutputField(
        desc='The paper\'s status, either "accepted" or "rejected"'
    )


class CoTModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = AChainOfThought(PeerReadSignature)

    async def forward(self, abstract: str):
        return await self.prog(abstract=abstract)

    async def __call__(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)


class ZeroShotModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = APredict(PeerReadSignature)

    async def forward(self, **kwargs):
        return await self.prog(**kwargs)

    async def __call__(self, *args, **kwargs):
        return await self.forward(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--n", type=int, default=-1)
    args = parser.parse_args()

    # configure LM
    llama = dspy.AOpenAI(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        max_tokens=512,
        temperature=0.0,
        api_base=VLLM_API_BASE,
        api_key=VLLM_API_KEY,
        model_type="chat",
        seed=2024,
    )
    dspy.settings.configure(lm=llama)

    # load ds
    ds = adapter.PeerReadDspy(trainsize=0.1, devsize=0.05, inputs=["abstract"])
    test_ds = ds.test[: args.n] if args.n != -1 else ds.test
    print(
        f"Train set size: {len(ds.train)}, dev set size: {len(ds.dev)}, test set size: {len(ds.test)}"
    )

    # get all tokens and count them
    tokenizer: Tokenizer = Tokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct"
    )
    all_encodings = tokenizer.encode_batch(
        list(map(lambda ex: ex["abstract"], test_ds))
    )
    assert len(all_encodings) == len(test_ds)

    # now sum up lengths
    total_length = sum(len(x.tokens) for x in all_encodings)
    print("Total tokens in test dataset:", total_length)

    # prompter = dspy.teleprompt.KNNFewShot(KNN, 5, ds.train)
    # prompter = dspy.teleprompt.BootstrapFewShot(adapter.peerread_metric, max_labeled_demos=5, max_bootstrapped_demos=0)
    # prompter = dspy.teleprompt.MIPRO(
    #     adapter.peerread_metric, prompt_model=llama, task_model=llama
    # )
    module = ZeroShotModule() if not args.cot else CoTModule()
    # optimized_program = prompter.compile(
    #     student=m,
    #     trainset=ds.train,
    #     num_trials=10,
    #     max_bootstrapped_demos=1,
    #     max_labeled_demos=5,
    #     eval_kwargs={"num_threads": 3},
    # )

    # optimized_program.save("output_program2")

    # run experiment
    score, outputs, results = asyncio.run(
        adapter.run_experiment(module=module, ds=test_ds, n_threads=args.concurrency)
    )
    print(llama.inspect_history())
