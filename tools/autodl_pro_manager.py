#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoDL Pro 无状态实例与本地 IDE 自动化连接管理器
支持功能：
1. 实例状态查询 (status / list)
2. 实例一键开机 (start / up) 并轮询就绪
3. 实例一键关机 (stop / down)
4. 获取动态 SSH 节点与端口，自动注入本地 ~/.ssh/config (Host autodl-pro)
5. 实现本地 IDE (VSCode / Antigravity IDE) Remote-SSH 免密/免配置一键直连
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# 适配 Windows 控制台编码
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_API_HOST = "https://api.autodl.com"
DEFAULT_CONFIG_NAMES = [
    "autodl_config.json",
    ".autodl_config.json",
    "tools/autodl_config.json",
    "../autodl_config.json"
]

def load_config():
    """读取本地配置文件或环境变量"""
    config = {
        "api_token": os.environ.get("AUTODL_API_TOKEN", ""),
        "instance_uuid": os.environ.get("AUTODL_INSTANCE_UUID", ""),
        "instance_name_filter": "pro",
        "ssh_host_alias": "autodl-pro",
        "identity_file": "~/.ssh/id_rsa",
        "region": "west-D",
        "fs_mount_path": "/root/autodl-fs"
    }

    found_file = None
    for name in DEFAULT_CONFIG_NAMES:
        p = Path(name)
        if p.exists() and p.is_file():
            found_file = p
            break
        script_dir = Path(__file__).resolve().parent
        candidate = script_dir / name
        if candidate.exists() and candidate.is_file():
            found_file = candidate
            break

    if found_file:
        try:
            with open(found_file, "r", encoding="utf-8") as f:
                user_conf = json.load(f)
                config.update({k: v for k, v in user_conf.items() if v is not None and v != ""})
        except Exception as e:
            print(f"[WARN] 读取配置文件 {found_file} 失败: {e}")

    return config

def api_request(token, endpoint, method="POST", data=None):
    """发送 HTTP 请求到 AutoDL 开放接口"""
    if not token or token.startswith("YOUR_"):
        raise ValueError(
            "未检测到有效的 AutoDL API Token！\n"
            "请在 tools/autodl_config.json 中配置 'api_token'，\n"
            "或设置环境变量 export AUTODL_API_TOKEN='your_token'"
        )

    url = f"{BASE_API_HOST}{endpoint}"
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    headers = {
        "Authorization": token,
        "Content-Type": "application/json",
        "User-Agent": "AutoDL-Pro-Manager/1.0"
    }

    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            res_json = json.loads(raw)
            if res_json.get("code") != "Success":
                msg = res_json.get("msg") or res_json.get("message") or str(res_json)
                raise RuntimeError(f"API 请求失败 [{endpoint}]: {msg}")
            return res_json.get("data")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} 错误: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"网络连接错误: {e.reason}")

def get_pro_instance_list(token):
    """获取所有 Pro 容器实例列表"""
    data = api_request(token, "/api/v1/dev/instance/pro/list", method="POST", data={"page_index": 1, "page_size": 20})
    if isinstance(data, dict) and "list" in data:
        return data["list"]
    return []

def get_pro_snapshot(token, uuid):
    """获取指定 Pro 实例快照详情 (含 SSH 动态端口与 Host)"""
    endpoint = f"/api/v1/dev/instance/pro/snapshot?instance_uuid={uuid}"
    return api_request(token, endpoint, method="GET")

def get_pro_status(token, uuid):
    """获取指定 Pro 实例运行状态"""
    endpoint = f"/api/v1/dev/instance/pro/status?instance_uuid={uuid}"
    return api_request(token, endpoint, method="GET")

def update_ssh_config(host_alias, ssh_host, ssh_port, identity_file="~/.ssh/id_rsa"):
    """
    修改本地 ~/.ssh/config，精准维护 Host <host_alias>
    """
    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    config_path = ssh_dir / "config"

    lines = []
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_lines = []
    inside_target_host = False
    replaced = False

    new_block = [
        f"Host {host_alias}\n",
        f"    HostName {ssh_host}\n",
        f"    Port {ssh_port}\n",
        f"    User root\n",
        f"    IdentityFile {identity_file}\n",
        f"    StrictHostKeyChecking no\n",
        f"    UserKnownHostsFile /dev/null\n",
        f"    ServerAliveInterval 60\n",
        f"    ServerAliveCountMax 5\n"
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("Host ") or stripped.startswith("host "):
            parts = stripped.split()
            current_alias = parts[1] if len(parts) > 1 else ""
            if current_alias == host_alias:
                inside_target_host = True
                new_lines.extend(new_block)
                replaced = True
                i += 1
                while i < len(lines):
                    next_strip = lines[i].strip()
                    if next_strip.startswith("Host ") or next_strip.startswith("host "):
                        inside_target_host = False
                        break
                    i += 1
                continue
            else:
                inside_target_host = False

        if not inside_target_host:
            new_lines.append(line)
        i += 1

    if not replaced:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append("\n# AutoDL Pro Auto-Config\n")
        new_lines.extend(new_block)

    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"[OK] 成功更新本地 SSH 配置文件: {config_path}")
    print(f"     [Host]     : {host_alias}")
    print(f"     [HostName] : {ssh_host}")
    print(f"     [Port]     : {ssh_port}")
    print(f"     [User]     : root")

def resolve_target_uuid(instances, config):
    """根据 uuid 或筛选器确定目标机器"""
    target_uuid = config.get("instance_uuid")
    if target_uuid:
        for inst in instances:
            if inst.get("uuid") == target_uuid:
                return target_uuid
        # 如果列表中没搜到但指定了，仍然返回指定的 uuid
        return target_uuid

    if instances:
        return instances[0].get("uuid")

    raise RuntimeError("当前账号下没有任何 AutoDL Pro 实例！请先在控制台创建一个 Pro 实例。")

def cmd_start(config):
    """开机命令并注入 SSH 配置"""
    token = config["api_token"]
    print("[1/3] 正在查询 AutoDL Pro 实例列表...")
    instances = get_pro_instance_list(token)
    uuid = resolve_target_uuid(instances, config)

    status = get_pro_status(token, uuid)
    print(f"目标实例 UUID: {uuid}，当前运行状态: {status}")

    if status != "running":
        print(f"[2/3] 发送开机指令 (power_on) ...")
        api_request(token, "/api/v1/dev/instance/pro/power_on", method="POST", data={
            "instance_uuid": uuid,
            "payload": "gpu"
        })

        print("正在等待实例启动就绪 (轮询中)...")
        for retry in range(30):
            time.sleep(3)
            status = get_pro_status(token, uuid)
            print(f"  > 状态刷新 [{retry+1}/30]: {status}...")
            if status == "running":
                break
        else:
            raise TimeoutError("实例启动超时，请检查控制台机器状态或账户余额。")
    else:
        print("[2/3] 实例已处于 running 状态，无需重复开机。")

    print("[3/3] 正在获取实例快照详情与 SSH 动态路由...")
    snapshot = get_pro_snapshot(token, uuid)
    ssh_host = snapshot.get("proxy_host")
    ssh_port = snapshot.get("ssh_port")
    root_pwd = snapshot.get("root_password")

    if not ssh_host or not ssh_port:
        raise RuntimeError(f"未能获取到 SSH 连接节点与端口，快照返回: {snapshot}")

    update_ssh_config(
        host_alias=config.get("ssh_host_alias", "autodl-pro"),
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        identity_file=config.get("identity_file", "~/.ssh/id_rsa")
    )

    print("\n" + "=" * 65)
    print(f"🎉 准备就绪！现在可以在本地 IDE (VSCode / Antigravity) 中直接连接:")
    print(f"👉 远程 Host 名 : {config.get('ssh_host_alias', 'autodl-pro')}")
    print(f"👉 命令行直连   : ssh {config.get('ssh_host_alias', 'autodl-pro')}")
    if root_pwd:
        print(f"🔑 默认 root 密码 : {root_pwd} (如已配置 SSH 密钥则免密)")
    print("=" * 65 + "\n")

def cmd_stop(config):
    """关机命令以停止计费"""
    token = config["api_token"]
    instances = get_pro_instance_list(token)
    uuid = resolve_target_uuid(instances, config)

    status = get_pro_status(token, uuid)
    if status in ["shutdown", "stopped"]:
        print(f"[INFO] 实例 {uuid} 已经处于关机状态 ({status})，无需重复操作。")
        return

    print(f"正在向实例 {uuid} 发送关机指令 (power_off)...")
    api_request(token, "/api/v1/dev/instance/pro/power_off", method="POST", data={
        "instance_uuid": uuid
    })
    print(f"[OK] 关机指令已发出！GPU 计费已停止。")

def cmd_status(config):
    """查看所有 Pro 实例状态和详情"""
    token = config["api_token"]
    instances = get_pro_instance_list(token)
    if not instances:
        print("[INFO] 没有任何 Pro 实例。")
        return

    print(f"{'UUID':<20} {'地域':<12} {'显卡规格':<15} {'状态':<12} {'开机时间'}")
    print("-" * 80)
    for inst in instances:
        uuid = inst.get("uuid")
        region = inst.get("region_name") or inst.get("region_sign")
        gpu = inst.get("gpu_spec_uuid")
        status = inst.get("status")
        start_time = inst.get("started_at", {}).get("Time", "")[:19]
        print(f"{uuid:<20} {region:<12} {gpu:<15} {status:<12} {start_time}")

def cmd_update_ssh(config):
    """仅更新本地 SSH 配置，不改变机器运行状态"""
    token = config["api_token"]
    instances = get_pro_instance_list(token)
    uuid = resolve_target_uuid(instances, config)
    snapshot = get_pro_snapshot(token, uuid)
    ssh_host = snapshot.get("proxy_host")
    ssh_port = snapshot.get("ssh_port")
    root_pwd = snapshot.get("root_password")

    if not ssh_host or not ssh_port:
        raise RuntimeError("该实例尚未分配有效的 SSH 节点或尚未处于 running 状态。")

    update_ssh_config(
        host_alias=config.get("ssh_host_alias", "autodl-pro"),
        ssh_host=ssh_host,
        ssh_port=ssh_port,
        identity_file=config.get("identity_file", "~/.ssh/id_rsa")
    )
    if root_pwd:
        print(f"🔑 默认 root 密码 : {root_pwd}")

def main():
    parser = argparse.ArgumentParser(description="AutoDL Pro 无状态实例与本地 IDE 自动化连接管理器")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    subparsers.add_parser("start", aliases=["up", "on"], help="开机并自动更新本地 SSH Config")
    subparsers.add_parser("stop", aliases=["down", "off"], help="关机以停止扣费")
    subparsers.add_parser("status", aliases=["list"], help="列出实例和当前连接信息")
    subparsers.add_parser("update-ssh", help="仅更新当前运行实例的本地 SSH Config")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    config = load_config()

    cmd = args.command
    try:
        if cmd in ["start", "up", "on"]:
            cmd_start(config)
        elif cmd in ["stop", "down", "off"]:
            cmd_stop(config)
        elif cmd in ["status", "list"]:
            cmd_status(config)
        elif cmd == "update-ssh":
            cmd_update_ssh(config)
    except Exception as e:
        print(f"\n[ERROR] 操作失败: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
