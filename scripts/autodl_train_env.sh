#!/usr/bin/env bash
# ==============================================================================
# AutoDL Pro 极速训练与 I/O 隔离启动脚本
# 架构规范：
# 1. 代码来自 GitHub 纯代码仓库 (位于 /root/Ferret)
# 2. 数据集来自 AutoDL-FS 内网网络存储 (/root/autodl-fs/datasets/)
# 3. 极速 I/O：将 tar/zip 数据包解压到 /dev/shm (Linux 共享内存虚拟盘) 中供给 GPU 训练
# 4. 容灾持久化：Checkpoint 直接输出到 /root/autodl-fs/models/
# ==============================================================================

set -e

FS_ROOT="/root/autodl-fs"
DATASET_TAR="${FS_ROOT}/datasets/go_dataset.tar"
SHM_DATA_DIR="/dev/shm/go_dataset"
OUTPUT_DIR="${FS_ROOT}/models/ferret_go_checkpoints"

echo "=================================================="
echo "🚀 [AutoDL Pro] 开始执行训练前环境与 I/O 准备..."
echo "=================================================="

# 1. 检查 AutoDL-FS 挂载
if [ ! -d "${FS_ROOT}" ]; then
    echo "❌ 错误: 未检测到 AutoDL-FS 挂载目录: ${FS_ROOT}！"
    echo "请确认当前实例是否已在控制台或开机参数中挂载了同区 AutoDL-FS。"
    exit 1
fi

mkdir -p "${FS_ROOT}/datasets"
mkdir -p "${OUTPUT_DIR}"

# 2. 准备 /dev/shm 内存盘高速缓存
echo "📦 正在准备数据到 /dev/shm 极速内存盘..."
if [ -f "${DATASET_TAR}" ]; then
    if [ ! -d "${SHM_DATA_DIR}" ]; then
        echo "--> 发现数据集包: ${DATASET_TAR}，正在解压至 ${SHM_DATA_DIR}..."
        mkdir -p "${SHM_DATA_DIR}"
        tar -xf "${DATASET_TAR}" -C "/dev/shm/"
        echo "--> 解压完成！当前 /dev/shm 占用情况:"
        df -h /dev/shm
    else
        echo "--> /dev/shm 数据缓存已存在，跳过解压。"
    fi
else
    echo "⚠️ 提示: 未在 ${DATASET_TAR} 找到打包数据集。"
    echo "如果直接使用本地已存在数据或未打包数据集，请确保软链接或拷贝至 /dev/shm 避免小文件网络 I/O 瓶颈。"
fi

# 3. GitHub 代码一致性检查
echo "🔄 检查 GitHub 代码更新..."
if [ -d ".git" ]; then
    echo "--> 正在拉取远程最新代码: git pull origin main..."
    git pull origin main || echo "git pull 跳过或遇到未提交改动，继续训练流程"
fi

# 4. 执行训练命令示例 (实际命令根据微调参数配置)
echo "=================================================="
echo "🔥 启动 Ferret 围棋视觉微调训练..."
echo "📁 训练输入数据目录: ${SHM_DATA_DIR:-/root/autodl-fs/datasets/}"
echo "💾 Checkpoint 保存目录: ${OUTPUT_DIR}"
echo "=================================================="

# 下方可根据实际训练脚本配置环境变量与启动命令：
# python -m torch.distributed.run --nproc_per_node=1 ferret/train/train.py \
#     --model_name_or_path lmsys/vicuna-7b-v1.5 \
#     --data_path ${SHM_DATA_DIR}/go_all.json \
#     --image_folder ${SHM_DATA_DIR}/images \
#     --output_dir ${OUTPUT_DIR} \
#     ...
