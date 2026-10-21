import sys
import subprocess
import time
from pathlib import Path

def run_remote_file(local_path):
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    with open(local_path, "r", encoding="utf-8") as f:
        code = f.read()
    print(f"[Remote] Sending and executing {local_path} ...")
    p = subprocess.Popen(["ssh", "-n", "autodl-pro", "python3 -u"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    out, _ = p.communicate(input=code)
    print(out)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "tools/diag.py"
    run_remote_file(target)
