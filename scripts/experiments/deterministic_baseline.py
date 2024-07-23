import llm_critic.core.dspy_adapter as adapter
import dspy
from dataclasses import dataclass
from dspy import Prediction
import random


class PeerReadSignature(dspy.Signature):
    abstract = dspy.InputField()
    accepted = dspy.OutputField()


class TrueModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = PeerReadSignature

    def forward(self, abstract: str):
        return Prediction.from_completions([{"accepted": "True"}], self.signature)


class FalseModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = PeerReadSignature

    def forward(self, abstract: str):
        return Prediction.from_completions([{"accepted": "False"}], self.signature)


class RandomModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.signature = PeerReadSignature

    def forward(self, abstract: str):
        return Prediction.from_completions(
            [{"accepted": random.choice(["True", "False"])}], self.signature
        )


def run_true_baseline():
    ds = adapter.PeerReadDspy(trainsize=0.01, devsize=0.01, inputs=["abstract"])
    score, _, _ = adapter.run_experiment(TrueModule(), ds, n_threads=1)

    return score


def run_false_baseline():
    ds = adapter.PeerReadDspy(trainsize=0.01, devsize=0.01, inputs=["abstract"])
    score, _, _ = adapter.run_experiment(FalseModule(), ds, n_threads=1)

    return score


def run_random_baseline(trials: int = 25):
    ds = adapter.PeerReadDspy(trainsize=0.01, devsize=0.01, inputs=["abstract"])
    scores = []
    for _ in range(trials):
        score, _, _ = adapter.run_experiment(RandomModule(), ds, n_threads=1)
        scores.append(score)

    return sum(scores) / len(scores)


if __name__ == "__main__":
    print("True Result:", run_true_baseline())
    print("False Result:", run_false_baseline())
    print("Random Result:", run_random_baseline())
