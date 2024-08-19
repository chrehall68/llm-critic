"""
This script is for benchmarking performance of our data-parallel VLLM server.

After starting the data-parallel VLLM server, you can run

python3 benchmark.py --concurrency 1 --n 500

From experimenting on the AMD research cloud's 4xMI210 nodes, a single GPU can process roughly
256 concurrent requests, so to maximally utilize the data-parallel VLLM server, set concurrency to 256*{NUM_GPUS}
"""

from typing import List
import openai
import openai.types
import asyncio
from datetime import datetime
import datasets
import argparse
from tqdm import tqdm

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SEED = 2024


async def main(concurrency=1, n=500):
    client = openai.AsyncOpenAI(base_url="http://localhost:8001/v1/", api_key="*")
    ds = datasets.load_dataset("chreh/peer_read_neurips")["train"]
    ds = ds.select(range(n))
    e = asyncio.Event()
    completions: List[openai.types.Completion] = [None for _ in range(n)]
    progress = iter(tqdm(range(n)))

    async def task():
        while not e.is_set():
            try:
                idx = next(progress)
            except StopIteration:
                e.set()
                break
            completion = await client.completions.create(
                model=MODEL,
                prompt=ds[idx]["abstract"],
                max_tokens=300,
                temperature=0,
                seed=SEED,
            )
            completions[idx] = completion

    tasks = [task() for _ in range(concurrency)]

    start_time = datetime.now()
    await asyncio.gather(*tasks)
    try:
        next(progress)
    except StopIteration:
        pass
    end_time = datetime.now()

    print(end_time - start_time)
    total_tokens = 0
    for completion in completions:
        # print number of tokens used
        total_tokens += completion.usage.completion_tokens
        print(completion.choices[0].text)

    print(total_tokens)
    print("TPS: ", total_tokens / (end_time - start_time).total_seconds())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--n", type=int, default=500)
    args = parser.parse_args()
    asyncio.run(main(concurrency=args.concurrency, n=args.n))
