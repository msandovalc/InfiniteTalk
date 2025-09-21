# File: F:\Development\PyCharm\Projects\InfiniteTalk\wan\modules\wan22_model.py
import logging
import torch
from diffsynth import ModelManager

logging.basicConfig(level=logging.INFO)

class Wan22Model:
    """
    Model class for Wan2.2-S2V-14B, handling MoE and native audio conditioning.
    """
    def __init__(self, ckpt_dir, task="s2v-14b", device="cuda", quant="none"):
        """
        Initializes Wan2.2 model.
        :param ckpt_dir: Checkpoint path.
        :param task: Task type.
        :param device: Device.
        :param quant: Quantization type.
        """
        self.ckpt_dir = ckpt_dir
        self.task = task
        self.device = device
        self.quant = quant
        try:
            self.model_manager = ModelManager(ckpt_dir=self.ckpt_dir, task=self.task, device=self.device)
            self.model_manager = ModelManager(ckpt_dir=self.config.ckpt_dir, task=self.config.task, device=self.device)

            self.model = self.model_manager.load_model(convert_model_dtype=self.quant)
            logging.info(f"Wan2.2 model loaded with task {self.task} on {self.device}")
        except FileNotFoundError as e:
            logging.error(f"Checkpoint not found: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Model load error: {str(e)}")
            raise

    def forward(self, x, t, cond, audio_path=None):
        """
        Forward pass with MoE routing.
        :param x: Input tensor.
        :param t: Timestep.
        :param cond: Conditioning.
        :param audio_path: Audio path for S2V.
        """
        try:
            # MoE routing simulation
            routed_x = self.model.denoise(x, t, cond, audio_path=audio_path)
            logging.info("Forward pass completed")
            return routed_x
        except RuntimeError as e:
            logging.error(f"Forward runtime error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected forward error: {str(e)}")
            raise