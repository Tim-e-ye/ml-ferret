#!/bin/bash
# ====================================================================
# Ferret 围棋棋盘微调快速验证 (Dry-Run: 5 steps)
# ====================================================================

set -e

FS_ROOT="/root/autodl-fs"
MODEL_NAME_OR_PATH="${FS_ROOT}/models/vicuna-7b-v1.3"
PRETRAIN_PROJECTOR="${FS_ROOT}/models/llava-336px-pretrain-vicuna-7b-v1.3/mm_projector.bin"
VISION_TOWER="${FS_ROOT}/models/clip-vit-large-patch14-336"
DATASET_TAR="${FS_ROOT}/datasets/go_dataset.tar"

SHM_DATA_DIR="/dev/shm/go_dataset"
DATA_PATH="${SHM_DATA_DIR}/Datasets/go_train.json"
IMAGE_FOLDER="${SHM_DATA_DIR}"
OUTPUT_DIR="/root/autodl-fs/models/dry_run_output"

echo "================================================================"
echo "🧪 启动 Ferret 围棋微调 Dry-Run 验证 (Max Steps: 5)"
echo "底座语言模型: $MODEL_NAME_OR_PATH"
echo "底座视觉投影: $PRETRAIN_PROJECTOR"
echo "底座视觉编码: $VISION_TOWER"
echo "内存盘训练集: $DATA_PATH"
echo "================================================================"

# 1. 确保内存盘数据存在
if [ ! -d "${SHM_DATA_DIR}/Datasets" ]; then
    echo "📦 解压数据集至 /dev/shm..."
    mkdir -p "$SHM_DATA_DIR"
    tar -xf "$DATASET_TAR" -C "$SHM_DATA_DIR"
fi

mkdir -p "$OUTPUT_DIR"

# 2. 启动轻量验证 (仅 5 steps)
python -m torch.distributed.run --nproc_per_node=1 --master_port=25002 \
    ferret/train/train.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --version "ferret_go_v1" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --pretrain_mm_mlp_adapter "$PRETRAIN_PROJECTOR" \
    --add_go_grid_sampler True \
    --go_board_size 19 \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to "none"

echo "================================================================"
echo "🎉 Dry-Run 5 步验证顺利完成！梯度反向传播与 Loss 计算完全正常！"
echo "================================================================"
