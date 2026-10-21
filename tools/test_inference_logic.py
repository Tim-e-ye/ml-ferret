import os
import sys
import json
import torch
from PIL import Image
from transformers import AutoTokenizer

sys.path.insert(0, "/root/ml-ferret")
from ferret.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, GO_POSITION_TOKENS
from ferret.conversation import conv_templates
from ferret.mm_utils import tokenizer_image_token
from ferret.model.language_model.ferret_llama import FERRETLlamaForCausalLM, FERRETConfig
from ferret.train.train import ModelArguments
from peft import PeftModel

base_model_path = '/root/autodl-fs/models/vicuna-7b-v1.3'
ckpt_path = '/root/autodl-fs/models/ferret_go_checkpoints/checkpoint-200/mm_projector.bin'
stage2_lora_dir = '/root/autodl-fs/models/ferret_go_stage2_lora'

tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
model_args = ModelArguments(
    model_name_or_path=base_model_path,
    version='ferret_go_v1',
    vision_tower='/root/autodl-fs/models/clip-vit-large-patch14-336',
    add_go_grid_sampler=True,
    go_board_size=19,
    tune_mm_mlp_adapter=True,
)
config = FERRETConfig.from_pretrained(base_model_path)
config.tune_mm_mlp_adapter = True
config.add_go_grid_sampler = True
config.go_board_size = 19
config.vision_tower = '/root/autodl-fs/models/clip-vit-large-patch14-336'

model = FERRETLlamaForCausalLM.from_pretrained(base_model_path, config=config, torch_dtype=torch.float16).cuda()
model.get_model().initialize_vision_modules(model_args=model_args, add_go_grid_sampler=True, go_board_size=19)
model.initialize_vision_tokenizer(model_args=model_args, tokenizer=tokenizer, add_go_grid_sampler=True)

weights = torch.load(ckpt_path, map_location='cpu')
proj_w = {k.replace('model.mm_projector.', ''): v for k, v in weights.items() if 'model.mm_projector' in k}
if proj_w:
    model.get_model().mm_projector.load_state_dict(proj_w)
sampler_w = {k.replace('model.go_grid_sampler.projector.', 'projector.'): v for k, v in weights.items() if 'go_grid_sampler' in k}
if sampler_w and hasattr(model.get_model(), 'go_grid_sampler'):
    model.get_model().go_grid_sampler.load_state_dict(sampler_w)

non_lora_path = os.path.join(stage2_lora_dir, 'non_lora_trainables.bin')
if os.path.exists(non_lora_path):
    non_lora_w = torch.load(non_lora_path, map_location='cpu')
    embed_w = {k.split('embed_tokens.')[1]: v for k, v in non_lora_w.items() if 'embed_tokens' in k}
    if embed_w:
        model.get_input_embeddings().load_state_dict(embed_w)
    lm_head_w = {k.split('lm_head.')[1]: v for k, v in non_lora_w.items() if 'lm_head' in k}
    if lm_head_w:
        model.get_output_embeddings().load_state_dict(lm_head_w)

model = PeftModel.from_pretrained(model, stage2_lora_dir)
model = model.merge_and_unload()
model.get_model().mm_projector.to(device='cuda', dtype=torch.float16)
if hasattr(model.get_model(), 'go_grid_sampler'):
    model.get_model().go_grid_sampler.to(device='cuda', dtype=torch.float16)
model = model.cuda().to(torch.float16).eval()

# Load 1 sample
val_data = json.load(open('/dev/shm/go_dataset/Datasets/go_val.json'))
sample = val_data[0]
image = Image.open('/dev/shm/go_dataset/' + sample['image']).convert('RGB')
img_proc = model.get_vision_tower().image_processor
img_tensor = img_proc.preprocess(image, return_tensors='pt', do_resize=True, do_center_crop=False, size={'height': 336, 'width': 336})['pixel_values'][0].unsqueeze(0).cuda().half()

board_bbox = sample.get('board_bbox', [0, 0, sample.get('image_w', 1), sample.get('image_h', 1)])
norm_bbox = [board_bbox[0]/float(sample['image_w']), board_bbox[1]/float(sample['image_h']), board_bbox[2]/float(sample['image_w']), board_bbox[3]/float(sample['image_h'])]

# Test 1: Forward with the full ground truth conversation to check teacher forcing loss and predictions!
print("=== TEST 1: Teacher Forcing Loss and Predictions on GT ===")
conv = conv_templates['ferret_go_v1'].copy()
conv.append_message(conv.roles[0], sample['conversations'][0]['value'])
conv.append_message(conv.roles[1], sample['conversations'][1]['value'])
full_prompt = conv.get_prompt()
full_input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

with torch.no_grad():
    # Run forward through model
    out = model(input_ids=full_input_ids, images=img_tensor, board_bboxes=[norm_bbox])
    pred_tokens = out.logits.argmax(dim=-1)[0]
    
    # Check predictions along the sequence
    print("Full input tokens length:", full_input_ids.shape[1])
    # Find where ASSISTANT starts
    assistant_str = conv.roles[1] + ":"
    # Let's decode the predictions in the assistant section
    text_out = tokenizer.decode(pred_tokens[-300:], skip_special_tokens=False)
    print("Decoded predictions snippet:", repr(text_out[:150]))
    gt_snippet = repr(sample['conversations'][1]['value'][:150])
    print("Ground truth snippet:      ", gt_snippet)

print("\n=== TEST 2: Inspect go_position_token_ids and Tokenizer ===")
go_ids = model.config.go_position_token_ids
print("Number of go tokens:", len(go_ids))
print("First 3 tokens:", [tokenizer.decode([i]) for i in go_ids[:3]])
print("A1 token ID:", tokenizer.convert_tokens_to_ids("<go_A1>"))
print("Vocabulary size in tokenizer:", len(tokenizer))
print("Model embed_tokens shape:", model.get_input_embeddings().weight.shape)
print("Model lm_head shape:", model.get_output_embeddings().weight.shape)
