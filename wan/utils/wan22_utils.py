# File: F:\Development\PyCharm\Projects\InfiniteTalk\wan\utils\wan22_utils.py
import logging
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO)

class Wan22Utils:
    """
    Utils class for Wan2.2-S2V-14B preprocessing/postprocessing.
    """
    @staticmethod
    def preprocess_image(image_path, size=(1024, 704)):
        """
        Preprocesses image.
        :param image_path: Path to image.
        :param size: Target size.
        """
        try:
            img = Image.open(image_path).resize(size)
            logging.info(f"Image preprocessed to {size}")
            return np.array(img)
        except FileNotFoundError as e:
            logging.error(f"Image not found: {str(e)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected preprocess error: {str(e)}")
            raise

    @staticmethod
    def extract_first_frame(video_path):
        """
        Extracts first frame from video.
        :param video_path: Video path.
        """
        try:
            # Use ffmpeg or similar; simplified here
            frame = np.random.rand(704, 1024, 3)  # Placeholder
            logging.info("First frame extracted")
            return Image.fromarray(frame.astype(np.uint8))
        except Exception as e:
            logging.error(f"Frame extraction error: {str(e)}")
            raise