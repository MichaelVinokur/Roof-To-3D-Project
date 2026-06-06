import subprocess
import sys
import os

PYTHON = sys.executable

STEPS = [
    ("Segmentation",   "segment.py"),
    ("Depth",          "depth.py"),
    ("Point Cloud",    "pointcloud.py"),
    ("Reconstruction", "reconstruct.py"),
]

def run_step(name, script):
    print(f"\n{'='*60}")
    print(f"STEP: {name}  ({script})")
    print('='*60)
    result = subprocess.run(
        [PYTHON, script],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print(f"\nERROR: {script} failed with exit code {result.returncode}. Stopping pipeline.")
        sys.exit(result.returncode)

for name, script in STEPS:
    run_step(name, script)

print(f"\n{'='*60}")
print("Pipeline complete!")
print('='*60)
