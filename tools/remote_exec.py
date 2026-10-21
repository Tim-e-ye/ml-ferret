import subprocess
import time
import sys

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def ssh_exec(cmd, timeout=120, max_retries=3, delay=4):
    """Executes a command on autodl-pro with retry and delay to satisfy gateway rate limits."""
    for attempt in range(1, max_retries + 1):
        time.sleep(delay)
        try:
            p = subprocess.Popen(
                ["ssh", "-n", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=10", "autodl-pro", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace"
            )
            out, err = p.communicate(timeout=timeout)
            if p.returncode == 0:
                return out, err, 0
            if "Connection closed" in err or "Connection reset" in err or p.returncode == 255:
                print(f"[Attempt {attempt}/{max_retries}] Gateway connection reset, retrying in {delay * attempt}s...")
                time.sleep(delay * attempt)
                continue
            return out, err, p.returncode
        except subprocess.TimeoutExpired:
            p.kill()
            print(f"[Attempt {attempt}/{max_retries}] SSH command timed out, retrying...")
            time.sleep(delay)
    return "", "Max retries reached", -1

if __name__ == "__main__":
    cmd = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "nvidia-smi"
    out, err, code = ssh_exec(cmd)
    if out:
        print(out)
    if err:
        print("STDERR:", err)
    sys.exit(code)
