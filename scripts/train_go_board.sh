#!/bin/bash
# ====================================================================
# Ferret 围棋棋盘特化训练启动脚本 (AutoDL 环境)
# ====================================================================

# 1. 基础路径配置 (根据 AutoDL 实际存放路径调整)
MODEL_NAME_OR_PATH="ferret-7b-v1-3"        # 基础预训练权重目录或 HuggingFace ID
DATA_PATH="./Datasets/go_all.json"         # 围棋训练数据 JSON 路径
IMAGE_FOLDER="./Datasets/captures"         # 原始围棋截图图片目录
OUTPUT_DIR="./checkpoints/ferret-go-7b"    # 训练输出与权重保存目录

# 2. 深度学习训练参数
BATCH_SIZE=4
GRAD_ACCUMULATION=4
LEARNING_RATE=2e-5
NUM_EPOCHS=3

echo "================================================================"
echo "开始 Ferret 围棋棋盘特化训练"
echo "基础模型: $MODEL_NAME_OR_PATH"
echo "训练数据: $DATA_PATH"
echo "截图目录: $IMAGE_FOLDER"
echo "保存路径: $OUTPUT_DIR"
echo "================================================================"

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
    --vision_tower "openai/clip-vit-large-patch14-336" \
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
