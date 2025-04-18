import dspy.datasets
import dspy.evaluate.evaluate
from ..data import load_dataset
import dspy
from typing import List


class PeerReadDspy:
    def __init__(
        self,
        trainsize: float = 0.1,
        devsize: float = 0.1,
        inputs: List[str] = ["abstract", "title"],
    ) -> None:
        dataset = load_dataset()

        # dataset has fields
        # abstract, title, accepted
        dataset.shuffle(2024)
        dataset = dataset.to_list()
        for entry in dataset:
            entry["status"] = "accepted" if entry["accepted"] else "rejected"

        trainset_end = int(len(dataset) * trainsize)
        devset_end = int(len(dataset) * (trainsize + devsize))

        self.train = [
            dspy.Example(**dataset[i]).with_inputs(*inputs) for i in range(trainset_end)
        ]
        self.dev = [
            dspy.Example(**dataset[i]).with_inputs(*inputs)
            for i in range(trainset_end, devset_end)
        ]
        self.test = [
            dspy.Example(**dataset[i]).with_inputs(*inputs)
            for i in range(devset_end, len(dataset))
        ]


def peerread_metric(
    y_true, y_pred, trace=None, trues=["accepted"], falses=["rejected"]
):
    y_pred_bool = None
    if any(x in y_pred.status.lower() for x in trues):
        y_pred_bool = True
    elif any(x in y_pred.status.lower() for x in falses):
        y_pred_bool = False

    return y_true.accepted == y_pred_bool


def run_experiment(module: dspy.Module, ds, n_threads: int = 1):
    evaluate = dspy.evaluate.Evaluate(
        devset=ds,
        metric=peerread_metric,
        num_threads=n_threads,
        display_progress=True,
    )
    return evaluate(module, return_outputs=True, return_all_scores=True)
