#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 ModelScope (阿里魔搭社区) 极速下载 Vicuna-7B-v1.3 权重
国内 CDN 直连，彻底规避海外 cas-bridge.xethub.hf.co 断流超时问题
"""

import os
import sys

try:
    from modelscope import snapshot_download
except ImportError:
    print("[INFO] 正在自动安装 modelscope...")
    os.system(f"{sys.executable} -m pip install -U modelscope")
    from modelscope import snapshot_download

target_dir = "/root/autodl-fs/models/vicuna-7b-v1.3"
os.makedirs(target_dir, exist_ok=True)

print("=" * 60)
print(f"🚀 开始通过 ModelScope 国内 CDN 极速拉取 lmsys/vicuna-7b-v1.3...")
print(f"📁 目标目录: {target_dir}")
print("=" * 60)

model_dir = snapshot_download(
    model_id="lmsys/vicuna-7b-v1.3",
    local_dir=target_dir
)

print("\n" + "=" * 60)
print(f"🎉 Vicuna-7B-v1.3 权重全部下载完毕！存放位置: {model_dir}")
print("=" * 60)
