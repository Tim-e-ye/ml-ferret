#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ferret 预训练底座权重下载器 (直接持久化到 AutoDL-FS)
目标：
1. lmsys/vicuna-7b-v1.3 (LLM 语言基座，约 13GB)
2. liuhaotian/llava-336px-pretrain-vicuna-7b-v1.3 (LLaVA 第一阶段预训练视觉投影头，约 50MB)
3. openai/clip-vit-large-patch14-336 (视觉编码器，约 1.7GB)
"""

import os
import sys
import time

# 启用 HF 国内镜像或官方直连
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("[INFO] 正在自动安装 huggingface_hub...")
    os.system(f"{sys.executable} -m pip install -U huggingface_hub")
    from huggingface_hub import snapshot_download

BASE_MODELS_DIR = "/root/autodl-fs/models"
os.makedirs(BASE_MODELS_DIR, exist_ok=True)

TARGETS = [
    {
        "repo_id": "liuhaotian/llava-336px-pretrain-vicuna-7b-v1.3",
        "local_dir": os.path.join(BASE_MODELS_DIR, "llava-336px-pretrain-vicuna-7b-v1.3"),
        "description": "LLaVA 第一阶段预训练视觉投影器权重"
    },
    {
        "repo_id": "openai/clip-vit-large-patch14-336",
        "local_dir": os.path.join(BASE_MODELS_DIR, "clip-vit-large-patch14-336"),
        "description": "OpenAI CLIP-ViT-L/14-336px 视觉编码器"
    },
    {
        "repo_id": "lmsys/vicuna-7b-v1.3",
        "local_dir": os.path.join(BASE_MODELS_DIR, "vicuna-7b-v1.3"),
        "description": "Vicuna-7B-v1.3 语言大模型底座"
    }
]

def main():
    print("=" * 60)
    print("🚀 开始下载 Ferret 预训练底座权重至 AutoDL-FS...")
    print(f"📁 目标存储路径: {BASE_MODELS_DIR}")
    print("=" * 60)

    for item in TARGETS:
        repo_id = item["repo_id"]
        local_dir = item["local_dir"]
        desc = item["description"]

        print(f"\n📥 [正在处理] {repo_id} ({desc})...")
        print(f"   目标目录: {local_dir}")

        for attempt in range(1, 4):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    max_workers=4
                )
                print(f"✅ [下载完成] {repo_id}")
                break
            except Exception as e:
                print(f"⚠️ [重试 {attempt}/3] 下载遇到错误: {e}")
                time.sleep(3)
        else:
            print(f"❌ [失败] {repo_id} 多次下载尝试均未成功，请检查网络后重试。")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("🎉 所有底座模型权重已成功下载并持久化至 AutoDL-FS！")
    print(f"您可以运行 'ls -lh {BASE_MODELS_DIR}' 查看已下载权重。")
    print("=" * 60)

if __name__ == "__main__":
    main()
