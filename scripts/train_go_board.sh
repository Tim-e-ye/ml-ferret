#!/bin/bash
# ====================================================================
# Ferret 围棋棋盘特化训练启动脚本 (AutoDL 环境)
# ====================================================================

# 1. 基础路径配置 (匹配 AutoDL-FS 持久化路径与 /dev/shm 内存加速盘)
FS_ROOT="/root/autodl-fs"
MODEL_NAME_OR_PATH="${FS_ROOT}/models/vicuna-7b-v1.3"
PRETRAIN_PROJECTOR="${FS_ROOT}/models/llava-336px-pretrain-vicuna-7b-v1.3/mm_projector.bin"
VISION_TOWER="${FS_ROOT}/models/clip-vit-large-patch14-336"
DATASET_TAR="${FS_ROOT}/datasets/go_dataset.tar"

SHM_DATA_DIR="/dev/shm/go_dataset"
DATA_PATH="${SHM_DATA_DIR}/Datasets/go_train.json"
IMAGE_FOLDER="${SHM_DATA_DIR}"
OUTPUT_DIR="${FS_ROOT}/models/ferret_go_checkpoints"

# 2. 深度学习训练参数 (针对 RTX 4090 24G 显存优化)
BATCH_SIZE=2
GRAD_ACCUMULATION=8
LEARNING_RATE=2e-5
NUM_EPOCHS=3

echo "================================================================"
echo "🚀 开始 Ferret 围棋棋盘特化训练"
echo "语言底座: $MODEL_NAME_OR_PATH"
echo "视觉投影器: $PRETRAIN_PROJECTOR"
echo "视觉编码器: $VISION_TOWER"
echo "训练数据: $DATA_PATH"
echo "图片目录: $IMAGE_FOLDER"
echo "输出目录: $OUTPUT_DIR"
echo "================================================================"

# 准备 /dev/shm 极速内存盘
if [ -f "$DATASET_TAR" ] && [ ! -d "$SHM_DATA_DIR" ]; then
    echo "📦 正在将数据集解压至 /dev/shm 极速内存盘..."
    mkdir -p "$SHM_DATA_DIR"
    tar -xf "$DATASET_TAR" -C "$SHM_DATA_DIR"
    echo "✅ /dev/shm 解压完成！"
fi

# 3. 运行前快速环境与架构验证
python scripts/test_go_model.py
if [ $? -ne 0 ]; then
    echo "模型组件测试失败，请检查 Python/PyTorch 依赖后再试！"
    exit 1
fi

# 4. 启动微调
# 注意：
#   --add_go_grid_sampler True: 启用 361 点网格采样器
#   --go_board_size 19: 19x19 规格
#   --version ferret_go_v1: 使用围棋专用对话模板
#   --tune_mm_mlp_adapter True: 第一阶段可仅微调投影头与采样器；全量微调可设为 False
python -m torch.distributed.run --nproc_per_node=1 --master_port=25001 \
    ferret/train/train.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --version "ferret_go_v1" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --pretrain_mm_mlp_adapter "$PRETRAIN_PROJECTOR" \
    --add_go_grid_sampler True \
    --go_board_size 19 \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "tensorboard"
