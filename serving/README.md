# Serving

This directory contains the code used to set up and serve meta-llama/Meta-Llama-3-8B-Instruct on the AMD research cluster using VLLM.

## Set up VLLM

Most of this comes from the [VLLM ROCm page](https://docs.vllm.ai/en/latest/getting_started/amd-installation.html), although there are a couple
of changes to make it work with the AMD research cloud.

```sh
cd $WORK
export LLM_CRITIC_LOCATION=$WORK/llm-critic/
mkdir Documents

# set rocm version
export ROCM_VERSION=6.1.2
module load rocm/${ROCM_VERSION}

# create/activate venv
python3 -m venv vllmvenv
source vllmvenv/bin/activate
pip install --upgrade pip

# download vllm repository
cd Documents
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Install PyTorch
pip install --no-cache-dir --pre torch==2.5.0.dev20240726 --index-url https://download.pytorch.org/whl/nightly/rocm6.1

# Build & install AMD SMI
pip install /opt/rocm-${ROCM_VERSION}/share/amd_smi

# Install dependencies
pip install cmake ninja wheel
pip install --upgrade numba scipy huggingface-hub[cli]
pip install "numpy<2"
pip install -r requirements-rocm.txt

# edit setup.py
python3 ${LLM_CRITIC_LOCATION}/serving/edit.py ${LLM_CRITIC_LOCATION} $WORK/Documents/vllm ${ROCM_VERSION}

# Build/install vLLM
python3 setup.py develop
```

## Start Data-Parallel Server

VLLM has options for pipeline parallel and tensor parallel serving. However, when serving these smaller models that fit
on a single GPU, tensor-parallel doesn't give as large improvements in efficiency as data parallel. Thus, we have a
minimal VLLM data-parallel server in `load_balancer.py`. You can run it manually by specifying the URLs of the remotes
that work should be balanced to. Alternatively, check out `vllm_script.sh` for usage on a 4xMI250 node.
