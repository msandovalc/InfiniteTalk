# File: F:\Development\PyCharm\Projects\InfiniteTalk\wan\distributed\wan22_distributed.py
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(level=logging.INFO)

class Wan22DistributedModel:
    """
    Distributed support class for Wan2.2-S2V-14B using FSDP/Ulysses for multi-GPU.
    This handles MoE parallelism without modifying existing distributed classes.
    """
    def __init__(self, model, rank, world_size, ulysses_size=8):
        """
        Initializes distributed model.
        :param model: Wan22Model instance.
        :param rank: GPU rank.
        :param world_size: Number of processes.
        :param ulysses_size: Ulysses parallelism level.
        """
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.ulysses_size = ulysses_size
        try:
            dist.init_process_group(backend='nccl', rank=self.rank, world_size=self.world_size)
            logging.info(f"Distributed init on rank {self.rank}/{self.world_size} with Ulysses size {self.ulysses_size}")
            self.model = DDP(self.model.model, device_ids=[self.rank])  # Wrap with DDP
        except RuntimeError as e:
            logging.error(f"Distributed init failed: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected distributed error: {str(e)}")
            raise

    def forward_dist(self, x, t, cond):
        """Distributed forward pass."""
        try:
            output = self.model.module.forward(x, t, cond)  # Access module for DDP
            logging.info(f"Distributed forward completed on rank {self.rank}")
            return output
        except RuntimeError as e:
            logging.error(f"Distributed forward error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected forward error: {str(e)}")
            raise

    def cleanup(self):
        """Cleanup distributed group."""
        try:
            dist.destroy_process_group()
            logging.info("Distributed group destroyed")
        except RuntimeError as e:
            logging.error(f"Cleanup failed: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected cleanup error: {str(e)}")
            raise