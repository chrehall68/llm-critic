"""
Runs the RoBERTA baseline experiment.

Example usage:

python3 scripts/experiments/baseline.py
"""

import llm_critic.core.baseline as baselines


if __name__ == "__main__":
    baselines.run_baseline()
