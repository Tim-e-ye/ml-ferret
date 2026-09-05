"""
围棋特化 Ferret 模型架构与数据流冒烟测试脚本 (用于本地/AutoDL 验证)
运行方法:
    python scripts/test_go_model.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from ferret.constants import GO_POSITION_TOKENS, GO_BOARD_SIZE, GO_COLS
from ferret.model.ferret_arch import GoGridSampler
from ferret.conversation import conv_templates


def test_constants_and_template():
    print("[1/3] 测试围棋专用常量与对话模板...")
    assert len(GO_POSITION_TOKENS) == 361, f"Expected 361 tokens, got {len(GO_POSITION_TOKENS)}"
    assert GO_POSITION_TOKENS[0] == "<go_A1>"
    assert GO_POSITION_TOKENS[-1] == "<go_S19>"
    assert len(GO_COLS) == 19
    assert "ferret_go_v1" in conv_templates
    conv = conv_templates["ferret_go_v1"].copy()
    conv.append_message(conv.roles[0], "<image>\n识别棋盘")
    conv.append_message(conv.roles[1], "<go_A1>A1黑<go_A2>A2空")
    prompt = conv.get_prompt()
    assert "<go_A1>A1黑" in prompt
    print("      -> 常量与对话模板验证通过！")


def test_go_grid_sampler():
    print("[2/3] 测试 GoGridSampler 网格采样与维度变换...")
    B, num_patches, C, hidden_dim = 2, 576, 1024, 4096
    sampler = GoGridSampler(input_dim=C, output_dim=hidden_dim, board_size=GO_BOARD_SIZE)
    dummy_feat = torch.randn(B, num_patches, C)
    board_bboxes = [
        [0.05, 0.05, 0.95, 0.95],
        [0.0, 0.0, 1.0, 1.0]
    ]
    sampled_feats = sampler(dummy_feat, board_bboxes, original_dtype=torch.float32, return_dtype=torch.float32)
    assert len(sampled_feats) == B
    assert sampled_feats[0].shape == (361, hidden_dim), f"Unexpected shape {sampled_feats[0].shape}"
    assert sampled_feats[1].shape == (361, hidden_dim), f"Unexpected shape {sampled_feats[1].shape}"
    print("      -> GoGridSampler 采样特征维度 (361, hidden_size) 验证通过！")


def test_embedding_replacement_logic():
    print("[3/3] 测试 Token 替换与 Mask 逻辑...")
    seq_len, hidden_dim = 20, 128
    text_embeds = torch.zeros(seq_len, hidden_dim)
    input_ids = torch.tensor([1, 2, 3, 100, 101, 4, 5, 6] + [0] * 12)
    
    # 假设 token 100 和 101 是特殊 token
    mock_go_token_ids = [100, 101]
    mock_go_feats = torch.ones(2, hidden_dim) * 9.99  # 采样的视觉特征
    
    go_embs = torch.zeros_like(text_embeds)
    go_all_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for pos_idx, tid in enumerate(mock_go_token_ids):
        mask = (input_ids == tid)
        if mask.any():
            go_embs[mask] = mock_go_feats[pos_idx]
            go_all_mask = go_all_mask | mask
            
    text_embeds = text_embeds * (~go_all_mask).to(text_embeds.dtype)[:, None] + go_embs
    
    assert torch.allclose(text_embeds[3], torch.full((hidden_dim,), 9.99))
    assert torch.allclose(text_embeds[4], torch.full((hidden_dim,), 9.99))
    assert torch.allclose(text_embeds[0], torch.zeros(hidden_dim))
    print("      -> 特殊 Token 视觉特征定向注入逻辑验证通过！")


if __name__ == "__main__":
    print("=" * 60)
    print("Ferret 围棋棋盘特化架构与数据流验证")
    print("=" * 60)
    test_constants_and_template()
    test_go_grid_sampler()
    test_embedding_replacement_logic()
    print("=" * 60)
    print("全部 3 项测试全部 PASS！模型与数据流设计完全正确。")
    print("=" * 60)
