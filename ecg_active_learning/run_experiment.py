import subprocess
import sys
from pathlib import Path

def main():
    args = sys.argv[1:]

    # Handle both being in root or being inside ecg_active_learning/
    current_dir = Path.cwd()
    if (current_dir / "src" / "experiment.py").exists():
        script_path = "src/experiment.py"
    else:
        script_path = "ecg_active_learning/src/experiment.py"

    cmd = ["python", script_path] + args
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
