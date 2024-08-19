"""
Looks for the line 
```
        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)
```
and inserts the call to `specialize.py` before it
"""

import argparse

SEARCH_STRING = (
    """        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)"""
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("llm_critic_path", type=str)
    parser.add_argument("vllm_path", type=str)
    parser.add_argument("version", type=str)
    args = parser.parse_args()

    # open setup.py
    with open(args.vllm_path + "/setup.py", "r") as f:
        setup = f.read()
    # divide into before and after
    before = setup[: setup.index(SEARCH_STRING)]
    insertion = f"""        os.system("python3 {args.llm_critic_path}/serving/specialize.py {args.vllm_path}/build {args.version}")\n"""
    after = setup[setup.index(SEARCH_STRING) :]

    # write new setup.py
    with open(args.vllm_path + "/setup.py", "w") as f:
        f.write(before + insertion + after)
