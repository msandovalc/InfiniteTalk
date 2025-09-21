# File: F:\Development\PyCharm\Projects\InfiniteTalk\wan\wan22.py
import logging
from .configs.wan22_config import Wan22Config

logging.basicConfig(level=logging.INFO)

class Wan22Initializer:
    """
    Root initializer class for Wan2.2-S2V-14B support.
    This class sets up the environment and logs initialization without modifying existing Wan classes.
    """
    def __init__(self, ckpt_dir, task="s2v-14b"):
        """
        Initializes Wan2.2 environment.
        :param ckpt_dir: Path to Wan2.2 checkpoint.
        :param task: Task type, default 's2v-14b'.
        """
        self.ckpt_dir = ckpt_dir
        self.task = task
        self.config = Wan22Config(ckpt_dir=ckpt_dir, resolution="1024*704").get_config()

        try:
            logging.info(f"Initializing Wan2.2 with checkpoint: {self.ckpt_dir} and task: {self.task}")
            # Simulate environment setup (e.g., check diffsynth version)
            from diffsynth import __version__ as diffsynth_version
            if diffsynth_version < '0.2.0':
                raise ValueError("diffsynth version must be >=0.2.0 for Wan2.2")
            logging.info("Wan2.2 environment initialized successfully")
        except ImportError as e:
            logging.error(f"Failed to import diffsynth: {str(e)}")
            raise
        except ValueError as e:
            logging.error(f"Initialization error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during initialization: {str(e)}")
            raise

    def get_model(self):
        """Returns a Wan22Model instance."""
        try:
            from .modules.wan22_model import Wan22Model
            logging.info("Creating Wan22Model instance")
            return Wan22Model(self.ckpt_dir, self.task)
        except ImportError as e:
            logging.error(f"Failed to import Wan22Model: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Error creating model: {str(e)}")
            raise