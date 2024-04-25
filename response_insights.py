import pickle as pk
import os


def fn(d):
    accepting = set(filter(lambda a: "accept" in a.lower(), d.keys()))
    rejecting = set(filter(lambda a: "reject" in a.lower(), d.keys()))
    both = set(
        filter(lambda a: "accept" in a.lower() and "reject" in a.lower(), d.keys())
    )
    print(
        f"Unique responses: accepting {len(accepting)}, rejecting {len(rejecting)}, both {len(both)}, total unique responses {len(d.keys())}"
    )
    total_accept = sum(len(d[key]) for key in accepting)
    total_reject = sum(len(d[key]) for key in rejecting)
    total_neither = sum(
        len(d[key]) for key in d if key not in accepting and key not in rejecting
    )
    print(
        f"total accept {total_accept}, total reject {total_reject}, total neither {total_neither}"
    )


if __name__ == "__main__":
    data_dir = "./critic/data/"
    for file in os.listdir(data_dir):
        if not file.endswith(".pk"):
            continue
        responses = pk.loads(open(data_dir + file, "rb").read())
        print(file)
        fn(responses)
