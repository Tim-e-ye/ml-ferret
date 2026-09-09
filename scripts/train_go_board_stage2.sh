#!/bin/bash
# ====================================================================
# Ferret 围棋棋盘特化训练 Stage 2: LoRA 联合微调
# ====================================================================
# 继承 Stage 1 已训练好的视觉投影层和网格采样器权重，
# 对 Vicuna-7B 语言模型的注意力层挂载 LoRA，
# 同时解冻 embed_tokens 与 lm_head，
# 让 361 个 Go 特殊 Token 获得真正的输入嵌入和输出概率。
# ====================================================================

# 1. 基础路径配置
FS_ROOT="/root/autodl-fs"
MODEL_NAME_OR_PATH="${FS_ROOT}/models/vicuna-7b-v1.3"
STAGE1_PROJECTOR="${FS_ROOT}/models/ferret_go_checkpoints/checkpoint-200/mm_projector.bin"
VISION_TOWER="${FS_ROOT}/models/clip-vit-large-patch14-336"
DATASET_TAR="${FS_ROOT}/datasets/go_dataset.tar"

SHM_DATA_DIR="/dev/shm/go_dataset"
DATA_PATH="${SHM_DATA_DIR}/Datasets/go_train.json"
IMAGE_FOLDER="${SHM_DATA_DIR}"
OUTPUT_DIR="${FS_ROOT}/models/ferret_go_stage2_lora"

# 2. LoRA 超参数
LORA_R=128
LORA_ALPHA=256

# 3. 训练超参数 (RTX 4090 48GB 优化)
BATCH_SIZE=2
GRAD_ACCUMULATION=8
LEARNING_RATE=2e-4
NUM_EPOCHS=5

echo "================================================================"
echo "🚀 开始 Ferret 围棋棋盘特化训练 — Stage 2 (LoRA + 解冻词表)"
echo "语言底座: $MODEL_NAME_OR_PATH"
echo "Stage 1 投影器权重: $STAGE1_PROJECTOR"
echo "视觉编码器: $VISION_TOWER"
echo "LoRA rank: $LORA_R, alpha: $LORA_ALPHA"
echo "训练数据: $DATA_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "================================================================"

# 准备 /dev/shm 极速内存盘
if [ -f "$DATASET_TAR" ] && [ ! -d "${SHM_DATA_DIR}/Datasets" ]; then
    echo "📦 正在将数据集解压至 /dev/shm 极速内存盘..."
    mkdir -p "$SHM_DATA_DIR"
    tar -xf "$DATASET_TAR" -C "$SHM_DATA_DIR"
    echo "✅ /dev/shm 解压完成！"
fi

# 4. 启动 Stage 2 微调
# 关键区别 vs Stage 1:
#   --tune_mm_mlp_adapter False: 不再冻结全模型（Stage 1 用此开关冻结 backbone）
#   --lora_enable True: 启用 LoRA 微调 LLM 注意力层
#   --unfreeze_embed_lm_head True: 解冻 embed_tokens 和 lm_head
#   --pretrain_mm_mlp_adapter: 加载 Stage 1 训练好的 projector + sampler 权重
python -m torch.distributed.run --nproc_per_node=1 --master_port=25001 \
    ferret/train/train.py \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --version "ferret_go_v1" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --pretrain_mm_mlp_adapter "$STAGE1_PROJECTOR" \
    --add_go_grid_sampler True \
    --go_board_size 19 \
    --tune_mm_mlp_adapter False \
    --lora_enable True \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout 0.05 \
    --unfreeze_embed_lm_head True \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps $GRAD_ACCUMULATION \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "tensorboard"

echo "================================================================"
echo "🎉 Ferret 围棋 Stage 2 LoRA 微调训练完成！"
echo "正在执行磁盘同步并在 5 秒后安全关机..."
echo "================================================================"
sync
sleep 5
/usr/bin/shutdown
