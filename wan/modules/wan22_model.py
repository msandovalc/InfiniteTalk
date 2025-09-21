
# File: /kaggle/working/InfiniteTalk/wan/modules/wan22_model.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import logging
import torch
from torch.amp import autocast
from safetensors.torch import load_file
from optimum.quanto import quantize, qint8
from .multitalk_model import WanModel
from ..wan_lora import WanLoraWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelManager:
    """Custom ModelManager for Wan2.2-S2V-14B, avoiding diffsynth dependencies."""
    def __init__(self, ckpt_dir, task, device, quant=None):
        self.ckpt_dir = ckpt_dir
        self.task = task
        self.device = device
        self.quant = quant
        try:
            self.model = WanModel(
                weight_init=False,
                checkpoint_path=f"{ckpt_dir}/diffusion_pytorch_model.safetensors",
                device=device,
                dtype=torch.float16 if quant == "fp8" else torch.float32
            )
            if quant == "qint8":
                quantize(self.model, weights=qint8)
                logging.info("Model quantized to qint8")
            logging.info(f"ModelManager initialized for task {task}")
        except Exception as e:
            logging.error(f"ModelManager initialization error: {str(e)}")
            raise

    def load_model(self):
        """Load the model."""
        try:
            return self.model
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

class Wan22Model:
    """
    Model class for Wan2.2-S2V-14B, bypassing diffsynth.
    """
    def __init__(self, ckpt_dir, task="s2v-14b", device="cuda", quant=None, lora_dir=None, lora_scale=1.0):
        self.ckpt_dir = ckpt_dir
        self.task = task
        self.device = device
        self.quant = quant
        self.lora_dir = lora_dir
        self.lora_scale = lora_scale
        try:
            self.model_manager = ModelManager(ckpt_dir=ckpt_dir, task=task, device=device, quant=quant)
            self.model = self.model_manager.load_model()
            if lora_dir:
                lora_wrapper = WanLoraWrapper(self.model)
                lora_name = lora_wrapper.load_lora(lora_dir)
                lora_wrapper.apply_lora(lora_name, lora_scale, param_dtype=torch.float16 if quant == "fp8" else torch.float32, device=device)
                logging.info(f"LoRA applied from {lora_dir}")
            logging.info(f"Wan22Model initialized for task {task}")
        except Exception as e:
            logging.error(f"Wan22Model initialization error: {str(e)}")
            raise

    def forward(self, x, t, cond):
        """
        Forward pass with autocast.
        """
        try:
            with autocast('cuda', enabled=self.quant == "fp8"):
                output = self.model.forward(x, t, cond)
                logging.info("Forward pass completed")
                return output
        except Exception as e:
            logging.error(f"Forward error: {str(e)}")
            raise

    def generate(self, image_path, audio_path, num_steps, mode):
        """
        Generate video (placeholder, integrate with pipeline).
        """
        try:
            with autocast('cuda', enabled=self.quant == "fp8"):
                # Placeholder logic; integrate with actual generation
                output = self.model.denoise(image_path, audio_path, num_steps, mode)
                logging.info("Generation completed")
                return output
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            raise
