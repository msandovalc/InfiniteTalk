# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
from easydict import EasyDict

from .shared_config import wan_shared_cfg

#------------------------ Wan I2V A14B ------------------------#

multitalk_A14B = EasyDict(__name__='Config: Wan MultiTalk AI2V A14B')
multitalk_A14B.update(wan_shared_cfg)
multitalk_A14B.sample_neg_prompt = 'bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards'

# T5
multitalk_A14B.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
multitalk_A14B.t5_tokenizer = 'google/umt5-xxl'

# CLIP (not separated in Wan2.2, integrated into T5)
multitalk_A14B.clip_model = None
multitalk_A14B.clip_dtype = torch.float16
multitalk_A14B.clip_checkpoint = None
multitalk_A14B.clip_tokenizer = None

# VAE (integrated into T5 checkpoint)
multitalk_A14B.vae_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
multitalk_A14B.vae_stride = (4, 16, 16)  # 16x16x4 compression for Wan2.2 VAE

# Transformer (MoE)
multitalk_A14B.patch_size = (1, 2, 2)
multitalk_A14B.dim = 6144
multitalk_A14B.ffn_dim = 16384
multitalk_A14B.freq_dim = 256
multitalk_A14B.num_heads = 48
multitalk_A14B.num_layers = 48
multitalk_A14B.num_experts = 2  # MoE with two experts
multitalk_A14B.active_experts = 1  # 1 active expert per step
multitalk_A14B.window_size = (-1, -1)
multitalk_A14B.qk_norm = True
multitalk_A14B.cross_attn_norm = True
multitalk_A14B.eps = 1e-6

# Support for 720P
multitalk_A14B.supported_sizes = ['infinitetalk-480', 'infinitetalk-720']