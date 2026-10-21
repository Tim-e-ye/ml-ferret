import time
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.autodl_pro_manager import load_config

def view_log(lines=35, follow=False):
    config = load_config()
    alias = config.get("ssh_host_alias", "autodl-pro")
    flag = "-f" if follow else f"-n {lines}"
    subprocess.run(["ssh", "-n", alias, f"if [ -f /root/eval.log ]; then tail {flag} /root/eval.log; else echo 'eval.log 尚未生成'; fi"])

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ["log", "tail"]:
        follow = "-f" in sys.argv or "--follow" in sys.argv
        view_log(lines=35, follow=follow)
        return

    config = load_config()
    alias = config.get("ssh_host_alias", "autodl-pro")
    
    print("⏳ 等待网络稳定...")
    time.sleep(3)
    
    cmd = 'nohup bash -l -c "cd /root/ml-ferret && python -u scripts/evaluate_go_board.py" </dev/null >/root/eval.log 2>&1 &'
    print(f"🚀 正在通过 SSH 在后台启动评测任务并重定向至 /root/eval.log...")
    subprocess.run(["ssh", "-n", alias, cmd])
    print("✅ 任务已提交至远端后台！")

if __name__ == "__main__":
    main()
