#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ferret-Go: 围棋特化多模态大模型 — 交互式 Web 演示系统
基于 Gradio 3.x，支持真实棋盘图片上传、端到端大模型推理识别、
交互式 SVG 19x19 棋盘高清渲染，以及一键导出 SGF 棋谱。
"""

import os
import sys
import re
import json
import time
import torch
from PIL import Image
import gradio as gr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, GO_POSITION_TOKENS, GO_COLS
from ferret.conversation import conv_templates
from ferret.mm_utils import tokenizer_image_token
from ferret.model.language_model.ferret_llama import FERRETLlamaForCausalLM, FERRETConfig
from ferret.train.train import ModelArguments
from peft import PeftModel

# -------------------------------------------------------------
# 1. 模型全局加载
# -------------------------------------------------------------
print("=" * 65)
print("🚀 正在初始化 Ferret-Go 围棋特化多模态大模型...")
print("=" * 65)

base_model_path = "/root/autodl-fs/models/vicuna-7b-v1.3"
ckpt_path = "/root/autodl-fs/models/ferret_go_checkpoints/checkpoint-200/mm_projector.bin"
stage2_lora_dir = "/root/autodl-fs/models/ferret_go_stage2_lora"

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)

model_args = ModelArguments(
    model_name_or_path=base_model_path,
    version="ferret_go_v1",
    vision_tower="/root/autodl-fs/models/clip-vit-large-patch14-336",
    add_go_grid_sampler=True,
    go_board_size=19,
    tune_mm_mlp_adapter=True,
)

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

model.get_model().initialize_vision_modules(
    model_args=model_args,
    add_go_grid_sampler=True,
    go_board_size=19
)
model.initialize_vision_tokenizer(
    model_args=model_args,
    tokenizer=tokenizer,
    add_go_grid_sampler=True
)

vision_tower = model.get_vision_tower()
vision_tower.to(dtype=torch.float16, device="cuda")
image_processor = vision_tower.image_processor

# 注入 Stage 1 投影权重与网格采样器
print(f"-> 注入 Stage 1 权重: {ckpt_path}")
weights = torch.load(ckpt_path, map_location="cpu")
proj_weights = {k.replace("model.mm_projector.", ""): v for k, v in weights.items() if "model.mm_projector" in k}
if proj_weights:
    model.get_model().mm_projector.load_state_dict(proj_weights)
sampler_weights = {k.replace("model.go_grid_sampler.projector.", "projector."): v for k, v in weights.items() if "go_grid_sampler" in k}
if sampler_weights and hasattr(model.get_model(), "go_grid_sampler"):
    model.get_model().go_grid_sampler.load_state_dict(sampler_weights)

# 注入 Stage 2 解冻词表
non_lora_path = os.path.join(stage2_lora_dir, "non_lora_trainables.bin")
if os.path.exists(non_lora_path):
    print(f"-> 注入 Stage 2 Go Token 词表嵌入: {non_lora_path}")
    non_lora_w = torch.load(non_lora_path, map_location="cpu")
    embed_w = {k.split("embed_tokens.")[1]: v for k, v in non_lora_w.items() if "embed_tokens" in k}
    if embed_w: model.get_input_embeddings().load_state_dict(embed_w)
    lm_head_w = {k.split("lm_head.")[1]: v for k, v in non_lora_w.items() if "lm_head" in k}
    if lm_head_w: model.get_output_embeddings().load_state_dict(lm_head_w)

# 注入 Stage 2 LoRA 并融合
if os.path.exists(os.path.join(stage2_lora_dir, "adapter_config.json")):
    print(f"-> 融合 Stage 2 LoRA 注意力层: {stage2_lora_dir}")
    model = PeftModel.from_pretrained(model, stage2_lora_dir)
    model = model.merge_and_unload()

model.get_model().mm_projector.to(device="cuda", dtype=torch.float16)
if hasattr(model.get_model(), "go_grid_sampler"):
    model.get_model().go_grid_sampler.to(device="cuda", dtype=torch.float16)
model = model.cuda().to(torch.float16).eval()

print("✅ 模型全量装载与权重融合就绪！随时可提供极速推理服务。")


# -------------------------------------------------------------
# 2. 状态解析与 SVG 棋盘渲染
# -------------------------------------------------------------
def parse_model_output(text):
    """从模型输出解析每个点位的黑白空状态"""
    state = {}
    pattern = re.compile(r'<go_([A-S]\d{1,2})>\s*\1\s*([黑白空])')
    for pos, status in pattern.findall(text):
        state[pos] = status
    
    if len(state) < 100:
        pattern2 = re.compile(r'<go_([A-S]\d{1,2})>\s*(?:[A-S]?\d{0,2})?\s*([黑白空])')
        for pos, status in pattern2.findall(text):
            state[pos] = status

    if len(state) < 100:
        pattern3 = re.compile(r'([A-S]\d{1,2})\s*[:：]?\s*([黑白空])')
        for pos, status in pattern3.findall(text):
            if pos not in state:
                state[pos] = status
    return state


def generate_sgf(state):
    """将棋盘状态转换为标准 SGF 格式"""
    black_stones = []
    white_stones = []
    for r in range(19):
        row_num = r + 1
        # 野狐规范: row 1 为顶部，对应 SGF 的 'a'
        row_char = chr(ord('a') + r)
        for c, col_letter in enumerate(GO_COLS):
            pos_name = f"{col_letter}{row_num}"
            val = state.get(pos_name, "空")
            col_char = chr(ord('a') + c)
            coord = f"{col_char}{row_char}"
            if val == "黑":
                black_stones.append(coord)
            elif val == "白":
                white_stones.append(coord)

    sgf = "(;GM[1]FF[4]SZ[19]AP[Ferret-Go-V1]"
    if black_stones:
        sgf += "AB" + "".join(f"[{c}]" for c in black_stones)
    if white_stones:
        sgf += "AW" + "".join(f"[{c}]" for c in white_stones)
    sgf += ")"
    return sgf


def render_svg_board(state):
    """
    生成高保真 19x19 围棋棋盘 SVG 矢量图
    - 实木底色质感
    - 标准 9 颗星位 (Hoshi)
    - 3D 拟真棋子阴影与高光
    """
    size = 580
    margin = 35
    cell = (size - 2 * margin) / 18.0  # 约 28.33 px

    svg = []
    svg.append(f'<svg width="100%" height="auto" viewBox="0 0 {size} {size}" style="max-width: 580px; display: block; margin: auto; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.3); background: #e2b772;">')
    
    # 渐变与滤镜定义
    svg.append('''
    <defs>
        <!-- 榧木棋盘纹理渐变 -->
        <linearGradient id="boardWood" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="#dfaf69"/>
            <stop offset="50%" stop-color="#d6a35a"/>
            <stop offset="100%" stop-color="#cb974c"/>
        </linearGradient>
        <!-- 黑子 3D 渐变 -->
        <radialGradient id="blackStone" cx="35%" cy="35%" r="65%">
            <stop offset="0%" stop-color="#555555"/>
            <stop offset="40%" stop-color="#222222"/>
            <stop offset="100%" stop-color="#050505"/>
        </radialGradient>
        <!-- 白子 3D 渐变 -->
        <radialGradient id="whiteStone" cx="35%" cy="35%" r="65%">
            <stop offset="0%" stop-color="#ffffff"/>
            <stop offset="60%" stop-color="#f0f0f0"/>
            <stop offset="100%" stop-color="#d4d4d4"/>
        </radialGradient>
        <!-- 棋子下落阴影 -->
        <filter id="stoneShadow" x="-20%" y="-20%" width="140%" height="140%">
            <feDropShadow dx="1.5" dy="2.5" stdDeviation="2" flood-opacity="0.35"/>
        </filter>
    </defs>
    ''')

    # 底板
    svg.append(f'<rect width="{size}" height="{size}" fill="url(#boardWood)" rx="8" stroke="#8b5e2b" stroke-width="3"/>')

    # 网格线
    for i in range(19):
        p = margin + i * cell
        # 横线
        svg.append(f'<line x1="{margin}" y1="{p:.1f}" x2="{size - margin}" y2="{p:.1f}" stroke="#4a2e12" stroke-width="1.2"/>')
        # 竖线
        svg.append(f'<line x1="{p:.1f}" y1="{margin}" x2="{p:.1f}" y2="{size - margin}" stroke="#4a2e12" stroke-width="1.2"/>')

    # 外边框加粗
    svg.append(f'<rect x="{margin}" y="{margin}" width="{size - 2*margin}" height="{size - 2*margin}" fill="none" stroke="#4a2e12" stroke-width="2.2"/>')

    # 9 个星位 (0-indexed: 3, 9, 15)
    star_indices = [3, 9, 15]
    for r in star_indices:
        for c in star_indices:
            cx = margin + c * cell
            cy = margin + r * cell
            svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.2" fill="#4a2e12"/>')

    # 坐标标签 (A-S, 1-19)
    font_style = 'font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; font-size: 11px; font-weight: bold; fill: #63421d; text-anchor: middle; dominant-baseline: middle;'
    for c, col_letter in enumerate(GO_COLS):
        x = margin + c * cell
        svg.append(f'<text x="{x:.1f}" y="{margin/2:.1f}" style="{font_style}">{col_letter}</text>')
        svg.append(f'<text x="{x:.1f}" y="{size - margin/2:.1f}" style="{font_style}">{col_letter}</text>')

    for r in range(19):
        row_num = r + 1
        y = margin + r * cell
        svg.append(f'<text x="{margin/2:.1f}" y="{y:.1f}" style="{font_style}">{row_num}</text>')
        svg.append(f'<text x="{size - margin/2:.1f}" y="{y:.1f}" style="{font_style}">{row_num}</text>')

    # 渲染棋子
    stone_radius = cell * 0.46
    for r in range(19):
        row_num = r + 1
        y = margin + r * cell
        for c, col_letter in enumerate(GO_COLS):
            x = margin + c * cell
            pos_name = f"{col_letter}{row_num}"
            val = state.get(pos_name, "空")
            if val == "黑":
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{stone_radius:.1f}" fill="url(#blackStone)" filter="url(#stoneShadow)"/>')
            elif val == "白":
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{stone_radius:.1f}" fill="url(#whiteStone)" filter="url(#stoneShadow)"/>')

    svg.append('</svg>')
    return "".join(svg)


# -------------------------------------------------------------
# 3. 核心推理函数
# -------------------------------------------------------------
def predict(input_image):
    if input_image is None:
        return "请先上传或选择一张围棋棋盘图片。", "", "", ""

    t_start = time.time()
    
    # 预处理输入图像
    image = input_image.convert("RGB")
    w, h = image.size
    
    img_tensor = image_processor.preprocess(
        image,
        return_tensors="pt",
        do_resize=True,
        do_center_crop=False,
        size={"height": 336, "width": 336}
    )["pixel_values"][0].unsqueeze(0).cuda().half()

    # 归一化棋盘区域 (默认全图检测，亦可传入精准检测框)
    norm_bbox = [0.0, 0.0, 1.0, 1.0]

    prompt = "<image>\n请识别这个围棋棋盘上所有棋子的位置。"
    conv = conv_templates["ferret_go_v1"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    # 执行模型自回归推理 (集成动态特征注入)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=img_tensor,
            board_bboxes=[norm_bbox],
            do_sample=False,
            max_new_tokens=1800,
            temperature=0.0
        )

    infer_time = time.time() - t_start
    output_str = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False)
    output_clean = output_str.replace("</s>", "").strip()

    # 解析棋盘点位状态
    state = parse_model_output(output_clean)
    
    # 统计子数
    black_count = sum(1 for v in state.values() if v == "黑")
    white_count = sum(1 for v in state.values() if v == "白")
    empty_count = sum(1 for v in state.values() if v == "空")
    total_parsed = len(state)

    # 渲染 SVG
    svg_html = render_svg_board(state)

    # 统计卡片 HTML
    stats_html = f'''
    <div style="display: flex; gap: 12px; margin-bottom: 12px; justify-content: space-around; font-family: sans-serif;">
        <div style="background: #1e293b; color: #f8fafc; padding: 12px 18px; border-radius: 8px; text-align: center; flex: 1; border-left: 4px solid #3b82f6;">
            <div style="font-size: 13px; color: #94a3b8;">⚫ 黑子 (Black)</div>
            <div style="font-size: 24px; font-weight: bold; margin-top: 4px;">{black_count}</div>
        </div>
        <div style="background: #1e293b; color: #f8fafc; padding: 12px 18px; border-radius: 8px; text-align: center; flex: 1; border-left: 4px solid #10b981;">
            <div style="font-size: 13px; color: #94a3b8;">⚪ 白子 (White)</div>
            <div style="font-size: 24px; font-weight: bold; margin-top: 4px;">{white_count}</div>
        </div>
        <div style="background: #1e293b; color: #f8fafc; padding: 12px 18px; border-radius: 8px; text-align: center; flex: 1; border-left: 4px solid #f59e0b;">
            <div style="font-size: 13px; color: #94a3b8;">🟩 空点 (Empty)</div>
            <div style="font-size: 24px; font-weight: bold; margin-top: 4px;">{empty_count}</div>
        </div>
        <div style="background: #1e293b; color: #f8fafc; padding: 12px 18px; border-radius: 8px; text-align: center; flex: 1; border-left: 4px solid #8b5cf6;">
            <div style="font-size: 13px; color: #94a3b8;">⚡ 推理耗时</div>
            <div style="font-size: 24px; font-weight: bold; margin-top: 4px;">{infer_time:.2f}s</div>
        </div>
    </div>
    '''

    # SGF 棋谱
    sgf_content = generate_sgf(state)

    return svg_html, stats_html, output_clean[:400] + ("..." if len(output_clean) > 400 else ""), sgf_content


# -------------------------------------------------------------
# 4. Gradio 页面构建
# -------------------------------------------------------------
custom_css = """
.gradio-container { max-width: 1250px !important; margin: auto !important; }
#title-header { text-align: center; margin-bottom: 20px; }
#board-output { display: flex; justify-content: center; align-items: center; }
"""

with gr.Blocks(title="Ferret-Go 围棋特化大模型体验系统", css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Column(elem_id="title-header"):
        gr.Markdown("""
        # ♟️ Ferret-Go: 围棋多模态特化大模型 — 视觉感知与智能复盘系统
        **基于 Ferret 架构 + 361 点位自适应网格采样器 (GoGridSampler) + Vicuna-7B LoRA 微调**
        """)

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 📸 1. 上传或选择棋盘图像")
            input_img = gr.Image(type="pil", label="真实对局棋盘截图")
            
            with gr.Row():
                btn_clear = gr.Button("🗑️ 清除", variant="secondary")
                btn_submit = gr.Button("🚀 开始智能识别", variant="primary")

            # 预置样例
            example_paths = [
                "/dev/shm/go_dataset/Datasets/Dataset1/fox_board-1_move_005.png",
                "/dev/shm/go_dataset/Datasets/Dataset1/fox_board-2_move_005.png",
                "/dev/shm/go_dataset/Datasets/Dataset1/fox_board-1_move_001.png"
            ]
            valid_examples = [p for p in example_paths if os.path.exists(p)]
            if valid_examples:
                gr.Examples(examples=valid_examples, inputs=input_img, label="💡 点击预设真实对局样例快速体验")

        with gr.Column(scale=7):
            gr.Markdown("### 🎯 2. 交互式复盘棋盘 (模型端到端识别输出)")
            stats_box = gr.HTML(label="数据统计")
            svg_board = gr.HTML(elem_id="board-output", label="高保真矢量棋盘")
            
            with gr.Accordion("📜 查看模型原始输出序列与 SGF 导出", open=False):
                raw_text_box = gr.Textbox(label="模型原始生成序列 (前 400 字符)", lines=3)
                sgf_box = gr.Textbox(label="导出 SGF 棋谱 (可直接复制至野狐/Sabaki/KataGo进行复盘分析)", lines=3)

    btn_submit.click(
        fn=predict,
        inputs=input_img,
        outputs=[svg_board, stats_box, raw_text_box, sgf_box]
    )
    btn_clear.click(lambda: (None, "", "", "", ""), outputs=[input_img, svg_board, stats_box, raw_text_box, sgf_box])

if __name__ == "__main__":
    # 优先绑定 6006 (AutoDL 控制台“自定义服务”原生支持一键在浏览器打开)
    # share=True 同时生成公网 gradio.live 链接
    demo.launch(server_name="0.0.0.0", server_port=6006, share=True)
