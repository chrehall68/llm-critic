# LLM-critic

This repository contains the code that was used for the work-in-progress paper `Leveraging Large Language Models (LLMs) for Effective
Research Paper Peer Review Support`

## Setup

To get started with this repository, simply run the following command in the root directory of this repository:

```bash
pip install -e .
```

This command installs the repository as a package in the current environment. This means that you can import the `llm_critic` package.

## Scripts

The `scripts` directory contains scripts that are used to run experiments. Each script is a separate python file that can be run from the command line.

## Running Experiments

The experiments require either a GPU or an LLM provider. Most times, it's cheaper to use a GPU than it is to use an LLM provider.
After all, our dataset contains millions and millions of tokens. If you want to use an LLM provider to handle that many tokens, you'll
definitely spend a lot of money and time (due to rate limits). Thus, we recommend using a GPU for all our experiments.

### Running on a GPU

We used RunPod to run our experiments. The steps are as follows:

- Create an account on [RunPod](https://runpod.ai)
- Create a pod with a GPU and at least 16GB of RAM, preferably 32GB
- run the following script to set up the environment and get a completion server running:

```sh
git clone https://github.com/chrehall68/llm-critic
cd llm-critic
pip install -e .
huggingface-cli login --token <YOUR_HUGGINGFACE_TOKEN>
pip install vllm  # --extra-index-url https://download.pytorch.org/whl/cu128  # required if using 50series GPU
vllm serve --max-model-len=4096 --gpu-memory-utilization=0.97 --dtype=bfloat16 meta-llama/Llama-3.1-8B-Instruct   # or any other huggingface model
```
