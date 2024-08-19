HIP_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8100 &
HIP_VISIBLE_DEVICES=1 vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8101 &
HIP_VISIBLE_DEVICES=2 vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8102 &
HIP_VISIBLE_DEVICES=3 vllm serve meta-llama/Meta-Llama-3-8B-Instruct --port 8103 &
python3 ./load_balancer.py --remotes http://localhost:8100 http://localhost:8101 http://localhost:8102 http://localhost:8103

# terminate all bg processes
# kill %1 %2 %3 %4