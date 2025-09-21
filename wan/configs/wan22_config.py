# File: F:\Development\PyCharm\Projects\InfiniteTalk\wan\configs\wan22_config.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import logging
from easydict import EasyDict
from .shared_config import wan_shared_cfg

logging.basicConfig(level=logging.INFO)

class Wan22Config:
    """
    Configuration class for Wan2.2-S2V-14B, aligning with existing wan/configs structure.
    Handles task-specific settings, MoE parameters, and integrates with shared config.
    """
    def __init__(self, ckpt_dir="Wan-AI/Wan2.2-S2V-14B", resolution="1024*704", moe_params=None):
        """
        Initializes Wan2.2 config.
        :param ckpt_dir: Path to Wan2.2 checkpoint, default 'Wan-AI/Wan2.2-S2V-14B'.
        :param resolution: Output resolution, default '1024*704'.
        :param moe_params: Dict for MoE, default {'num_experts': 27, 'active_experts': 14}.
        """
        self.config = EasyDict(__name__='Config: Wan2.2-S2V-14B')
        try:
            # Inherit shared config
            self.config.update(wan_shared_cfg)
            logging.info("Inherited shared config from wan_shared_cfg")

            # Task and checkpoint
            self.config.task = "s2v-14b"
            self.config.ckpt_dir = ckpt_dir
            logging.info(f"Set task: {self.config.task}, checkpoint: {self.config.ckpt_dir}")

            # Resolution
            self.config.supported_sizes = ['infinitetalk-480', 'infinitetalk-720', 'infinitetalk-1024']
            self.config.resolution = resolution
            width, height = map(int, resolution.split("*"))
            logging.info(f"Resolution set to {width}x{height}")

            # Negative prompt (reused from multitalk_A14B)
            self.config.sample_neg_prompt = (
                'bright tones, overexposed, static, blurred details, subtitles, style, works, '
                'paintings, images, static, overall gray, worst quality, low quality, '
                'JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, '
                'poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, '
                'still picture, messy background, three legs, many people in the background, '
                'walking backwards'
            )
            logging.info("Negative prompt set")

            # T5 and VAE (integrated, aligned with Wan2.2)
            self.config.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'  # Update if Wan2.2 uses different
            self.config.t5_tokenizer = 'google/umt5-xxl'
            self.config.vae_checkpoint = self.config.t5_checkpoint  # Integrated
            self.config.vae_stride = (4, 64, 64)  # Wan2.2 VAE: 64x compression
            logging.info("T5 and VAE configs set")

            # Transformer (MoE for Wan2.2)
            self.config.moe_params = moe_params or {'num_experts': 27, 'active_experts': 14}
            self.config.patch_size = (1, 2, 2)
            self.config.dim = 6144
            self.config.ffn_dim = 16384
            self.config.freq_dim = 256
            self.config.num_heads = 48
            self.config.num_layers = 48
            self.config.window_size = (-1, -1)
            self.config.qk_norm = True
            self.config.cross_attn_norm = True
            self.config.eps = 1e-6
            logging.info(f"MoE and transformer params set: {self.config.moe_params}")

            # Additional settings
            self.config.clip_model = None  # Integrated in T5
            self.config.clip_dtype = torch.float16
            self.config.clip_checkpoint = None
            self.config.clip_tokenizer = None
            logging.info("CLIP settings configured (integrated in T5)")

        except ValueError as e:
            logging.error(f"Invalid configuration format: {str(e)}")
            raise
        except FileNotFoundError as e:
            logging.error(f"Checkpoint path not found: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected config initialization error: {str(e)}")
            raise

    def get_resolution(self):
        """
        Returns resolution as (width, height) tuple.
        """
        try:
            return tuple(map(int, self.config.resolution.split("*")))
        except ValueError as e:
            logging.error(f"Error parsing resolution: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected resolution error: {str(e)}")
            raise

    def update_moe(self, new_params):
        """
        Updates MoE parameters.
        :param new_params: Dict with MoE settings (e.g., {'num_experts': 32}).
        """
        try:
            self.config.moe_params.update(new_params)
            logging.info(f"MoE params updated: {self.config.moe_params}")
        except AttributeError as e:
            logging.error(f"Invalid MoE params update: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected MoE update error: {str(e)}")
            raise

    def get_config(self):
        """
        Returns the full configuration as EasyDict.
        """
        try:
            logging.info("Returning full configuration")
            return self.config
        except Exception as e:
            logging.error(f"Error retrieving config: {str(e)}")
            raise