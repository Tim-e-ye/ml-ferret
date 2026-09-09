#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
围棋特化 Ferret 模型推理与效果评估脚本
用于加载微调后的 Checkpoint，输入真实棋盘图片，评估识别准确率并可视化棋盘识别结果
"""

import os
import sys
import json
import re
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, GO_POSITION_TOKENS, GO_COLS
from ferret.conversation import conv_templates
from ferret.model.builder import load_pretrained_model
from ferret.mm_utils import tokenizer_image_token, get_model_name_from_path
from ferret.model.language_model.ferret_llama import FERRETLlamaForCausalLM, FERRETConfig


def parse_go_state(text):
    """
    解析模型输出的字符串，形如: <go_A1>A1黑<go_B1>B1空...
    返回 dict: {"A1": "黑", "B1": "空", ...}
    """
    state = {}
    pattern = re.compile(r'<go_([A-S]\d{1,2})>\1([黑白空])')
    matches = pattern.findall(text)
    for pos, status in matches:
        state[pos] = status
    
    # 宽松正则后备匹配 (如果模型只输出了点位或类别)
    if len(state) < 100:
        pattern2 = re.compile(r'<go_([A-S]\d{1,2})>[A-S]?\d{0,2}([黑白空])')
        for pos, status in pattern2.findall(text):
            state[pos] = status
            
    return state


def evaluate():
    print("=" * 65)
    print("🎯 开始 Ferret 围棋棋盘特化模型效果评测")
    print("=" * 65)

    base_model_path = "/root/autodl-fs/models/vicuna-7b-v1.3"
    ckpt_path = "/root/autodl-fs/models/ferret_go_checkpoints/checkpoint-200/mm_projector.bin"
    val_json_path = "/dev/shm/go_dataset/Datasets/go_val.json"
    image_base_dir = "/dev/shm/go_dataset"

    if not os.path.exists(val_json_path):
        val_json_path = "Datasets/go_val.json"
        image_base_dir = "."

    print(f"1. 加载验证集数据: {val_json_path}")
    with open(val_json_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    print(f"   验证集样本数: {len(val_data)}")

    print(f"2. 加载基础语言模型与分词器: {base_model_path}")
    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    
    config = FERRETConfig.from_pretrained(base_model_path)
    config.tune_mm_mlp_adapter = True
    config.add_go_grid_sampler = True
    config.go_board_size = 19
    config.vision_tower = "/root/autodl-fs/models/clip-vit-large-patch14-336"

    model = FERRETLlamaForCausalLM.from_pretrained(
        base_model_path,
        config=config,
        torch_dtype=torch.float16
    ).cuda()

    model.initialize_vision_modules(
        config,
        add_go_grid_sampler=True,
        go_board_size=19
    )
    model.initialize_vision_tokenizer(config, tokenizer=tokenizer, add_go_grid_sampler=True)

    print(f"3. 注入微调好的围棋多模态权重: {ckpt_path}")
    weights = torch.load(ckpt_path, map_location="cpu")
    
    # 加载 mm_projector 与 go_grid_sampler
    proj_weights = {k.replace("model.mm_projector.", ""): v for k, v in weights.items() if "model.mm_projector" in k}
    if proj_weights:
        model.model.mm_projector.load_state_dict(proj_weights)
        print("   -> mm_projector 权重加载成功")

    sampler_weights = {k.replace("model.go_grid_sampler.projector.", "projector."): v for k, v in weights.items() if "go_grid_sampler" in k}
    if sampler_weights and hasattr(model.model, "go_grid_sampler"):
        model.model.go_grid_sampler.load_state_dict(sampler_weights)
        print("   -> go_grid_sampler 围棋网格采样器权重加载成功")

    model = model.cuda().eval()
    image_processor = model.get_vision_tower().image_processor

    print("\n4. 抽取验证集真实样本进行端到端推理测试...")
    # 测试前 3 个样本
    test_samples = val_data[:3]

    for idx, sample in enumerate(test_samples):
        img_rel = sample["image"]
        img_path = os.path.join(image_base_dir, img_rel)
        if not os.path.exists(img_path):
            img_path = img_rel

        print(f"\n" + "-" * 60)
        print(f"📸 测试样本 #{idx+1}: {os.path.basename(img_path)}")
        image = Image.open(img_path).convert("RGB")
        img_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].unsqueeze(0).cuda().half()

        # 归一化棋盘坐标
        board_bbox = sample.get("board_bbox", [0, 0, sample.get("image_w", 1), sample.get("image_h", 1)])
        img_w = float(sample.get("image_w", 1))
        img_h = float(sample.get("image_h", 1))
        norm_bbox = [board_bbox[0]/img_w, board_bbox[1]/img_h, board_bbox[2]/img_w, board_bbox[3]/img_h]

        prompt = "<image>\n请识别这个围棋棋盘上所有棋子的位置。"
        conv = conv_templates["ferret_go_v1"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        # 生成预测
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=img_tensor,
                board_bboxes=[norm_bbox],
                do_sample=False,
                max_new_tokens=1024,
                temperature=0.0
            )

        output_str = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
        output_clean = output_str.replace("</s>", "").strip()

        print("🔍 模型输出序列片段 (前 150 字符):")
        print(f"   {output_clean[:150]} ...")

        # 解析模型状态与 GT 状态
        pred_state = parse_go_state(output_clean)
        gt_state = sample.get("board_state", {})

        # 统计准确率
        correct = 0
        total_eval = 0
        black_correct, black_total = 0, 0
        white_correct, white_total = 0, 0
        empty_correct, empty_total = 0, 0

        status_mapping = {"black": "黑", "white": "白", "empty": "空"}

        for pos, gt_val in gt_state.items():
            gt_cn = status_mapping.get(gt_val, gt_val)
            pred_cn = pred_state.get(pos, "未识别")

            total_eval += 1
            if gt_cn == "黑":
                black_total += 1
                if pred_cn == "黑": black_correct += 1
            elif gt_cn == "白":
                white_total += 1
                if pred_cn == "白": white_correct += 1
            elif gt_cn == "空":
                empty_total += 1
                if pred_cn == "空": empty_correct += 1

            if pred_cn == gt_cn:
                correct += 1

        acc = (correct / total_eval * 100) if total_eval > 0 else 0
        b_acc = (black_correct / black_total * 100) if black_total > 0 else 0
        w_acc = (white_correct / white_total * 100) if white_total > 0 else 0
        e_acc = (empty_correct / empty_total * 100) if empty_total > 0 else 0

        print(f"\n📊 识别准确率评测统计 (共 {total_eval} 个交叉点位):")
        print(f"   - 全盘 361 点总体准确率 : {acc:.2f}% ({correct}/{total_eval})")
        print(f"   - 黑子识别准确率       : {b_acc:.2f}% ({black_correct}/{black_total})")
        print(f"   - 白子识别准确率       : {w_acc:.2f}% ({white_correct}/{white_total})")
        print(f"   - 空点识别准确率       : {e_acc:.2f}% ({empty_correct}/{empty_total})")

    print("\n" + "=" * 65)
    print("🎉 围棋模型验证集效果评测完成！")
    print("=" * 65)


if __name__ == "__main__":
    evaluate()
