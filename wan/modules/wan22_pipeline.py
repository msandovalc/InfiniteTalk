# File: F:\Development\PyCharm\Projects\InfiniteTalk\wan\modules\wan22_pipeline.py
import logging
from diffsynth import SDVideoPipeline

logging.basicConfig(level=logging.INFO)

class Wan22Pipeline:
    """
    Pipeline class for Wan2.2-S2V-14B inference.
    """
    def __init__(self, model, offload_model=True):
        """
        Initializes pipeline.
        :param model: Wan22Model instance.
        :param offload_model: Offload to CPU.
        """
        self.model = model
        try:
            self.pipeline = SDVideoPipeline.from_model_manager(self.model.model_manager, offload_model=offload_model)
            logging.info("Pipeline initialized")
        except ValueError as e:
            logging.error(f"Pipeline init error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected init error: {str(e)}")
            raise

    def generate(self, image_path, audio_path, num_steps=20, mode="streaming"):
        """
        Generates video.
        :param image_path: Input image.
        :param audio_path: Audio path.
        :param num_steps: Denoising steps.
        :param mode: Generation mode.
        """
        try:
            video = self.pipeline(image_path=image_path, audio_path=audio_path, num_inference_steps=num_steps, mode=mode)
            logging.info(f"Video generated in {mode} mode")
            return video
        except RuntimeError as e:
            logging.error(f"Generation runtime error: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected generation error: {str(e)}")
            raise