"""
This script changes all instances of /opt/rocm/ to /opt/rocm-{VERSION}/ in all build files
This is intended for changing all the cmake build files right before running the build.

Example usage:
python3 $WORK/Documents/specialize.py $WORK/Documents/vllm/build 6.1.2
"""

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("version", type=str)
args = parser.parse_args()
to_process = []

for root, dirs, files in os.walk(args.path):
    for file in files:
        if (
            file.endswith(".cmake")
            or file.endswith("Makefile")
            or file.endswith(".ninja")
        ):
            to_process.append(os.path.join(root, file))

for file in to_process:
    # get original timestamp
    time_accessed = os.path.getatime(file)
    time_modified = os.path.getmtime(file)
    tup = time_accessed, time_modified

    # now replace all instances of /opt/rocm/ with /opt/rocm-{VERSION}/
    with open(file, "r") as f:
        contents = f.read()
    contents = contents.replace("/opt/rocm/", "/opt/rocm-{}/".format(args.version))
    with open(file, "w") as f:
        f.write(contents)

    # update timestamp
    os.utime(file, tup)

print("done")
print("processed", to_process)
