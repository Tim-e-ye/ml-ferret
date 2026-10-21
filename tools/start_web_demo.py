#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ferret-Go Web 演示一键启动与端口映射工具
1. 检查并在远端后台启动 scripts/web_demo.py
2. 自动建立本地端口映射: localhost:7860 -> 远端 7860
3. 自动在本地浏览器中打开体验界面
"""

import sys
import time
import subprocess
import webbrowser
from pathlib import Path

# 配置 Windows 控制台输出编码
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def main():
    print("=" * 65)
    print("🌐 启动 Ferret-Go 围棋大模型 Web 交互演示系统")
    print("=" * 65)

    # 1. 检查远端是否已运行 web_demo.py
    print("📡 正在检查远端 GPU 服务状态...")
    check_cmd = "ps aux | grep web_demo.py | grep -v grep | awk '{print $2}'"
    p = subprocess.run(["ssh", "-n", "autodl-pro", check_cmd], capture_output=True, text=True, errors="replace")
    pid = p.stdout.strip()

    if not pid:
        print("🚀 正在远端 GPU 启动 Gradio Web 服务 (端口 7860)...")
        launch_cmd = 'cd /root/ml-ferret && nohup /root/miniconda3/bin/python -u scripts/web_demo.py </dev/null >/root/web_demo.log 2>&1 &'
        subprocess.run(["ssh", "-n", "autodl-pro", launch_cmd])
        print("⏳ 正在等待大模型加载与权重融合 (约需 30~50 秒)...")
        
        for i in range(25):
            time.sleep(3)
            # 检查端口 7860 是否监听
            p = subprocess.run(["ssh", "-n", "autodl-pro", "netstat -tuln | grep 7860"], capture_output=True, text=True, errors="replace")
            if "7860" in p.stdout:
                print("✅ 远端 Gradio 服务已成功监听 7860 端口！")
                break
            print(f"   [{(i+1)*3}s] 正在加载模型权重...", end="\r", flush=True)
        print()
    else:
        print(f"✅ 远端已在运行 Web 服务 (PID: {pid})")

    # 2. 建立 SSH 本地端口映射隧道: localhost:7860 -> autodl-pro:7860
    print("🔗 正在建立 SSH 本地端口转发通道 (http://localhost:7860)...")
    tunnel_cmd = ["ssh", "-N", "-L", "7860:localhost:7860", "autodl-pro"]
    tunnel_proc = subprocess.Popen(tunnel_cmd)

    time.sleep(2)
    url = "http://localhost:7860"
    print(f"\n🎉 演示系统已成功就绪！")
    print(f"👉 正在自动为您打开浏览器: {url}")
    print("💡 您可以在界面上选择样例图片或上传真实截图，点击【开始智能识别】体验端到端识别与棋盘复盘！")
    print("⚠️ 提示：若需退出体验，请在终端按 Ctrl+C 停止端口隧道。")
    print("-" * 65)

    try:
        webbrowser.open(url)
        tunnel_proc.wait()
    except KeyboardInterrupt:
        print("\n[INFO] 正在关闭本地 SSH 端口转发通道...")
        tunnel_proc.terminate()
        print("✅ 本地通道已安全关闭。")

if __name__ == "__main__":
    main()
