import dspy.datasets
import dspy.evaluate
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


def peerread_metric(y_true, y_pred, trace=None):
    trues = ["true", "yes", "accept"]
    falses = ["false", "no", "reject"]
    y_true_bool = None
    y_pred_bool = None
    if y_pred.accepted.lower() in trues:
        y_pred_bool = True
    elif y_pred.accepted.lower() in falses:
        y_pred_bool = False
    if str(y_true.accepted).lower() in trues:
        y_true_bool = True
    elif str(y_true.accepted).lower() in falses:
        y_true_bool = False

    assert y_true_bool is not None
    return y_pred_bool == y_true_bool


def run_experiment(module: dspy.Module, ds: PeerReadDspy, n_threads: int = 1):
    evaluate = dspy.evaluate.Evaluate(
        devset=ds.test,
        metric=peerread_metric,
        num_threads=n_threads,
        display_progress=True,
    )
    return evaluate(module, return_outputs=True, return_all_scores=True)
