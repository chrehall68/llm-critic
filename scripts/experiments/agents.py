"""
Runs the agent experiment

Example usage:

export OPENAI_API_KEY="*"
export OPENAI_BASE_URL="http://localhost:8000/v1/"
python3 scripts/experiments/agents.py --model meta-llama/Llama-3.1-8B --start 0 --end 100
"""

from llm_critic.agents import run_experiment
from llm_critic.data import load_dataset
from argparse import ArgumentParser
from collections import defaultdict
import pickle as pk

# custom parser because the arguments are different from the main script
parser = ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)
parser.add_argument("--num_concurrent", type=int, default=10)

if __name__ == "__main__":
    args = parser.parse_args()
    model = args.model
    start = args.start
    end = args.end
    num_concurrent = args.num_concurrent

    ds = load_dataset()
    end = len(ds) if end == -1 else end
    results = run_experiment(
        ds, start, end, defaultdict(lambda: model), model, num_concurrent
    )
    pk.dump(results, open(f"{model.replace('/', '_')}_{start}_{end}.pk", "wb"))
