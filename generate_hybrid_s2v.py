# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import logging
import torch
import os
import json
import ffmpeg
import numpy as np
from PIL import Image
from wan.wan22 import Wan22Initializer
from wan.configs.wan22_config import Wan22Config
from wan.modules.wan22_model import Wan22Model
from wan.modules.wan22_pipeline import Wan22Pipeline
from wan.utils.wan22_utils import Wan22Utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """
    Parse command-line arguments for Wan2.2-S2V-14B inference.
    """
    parser = argparse.ArgumentParser(description="Hybrid Wan2.2-S2V + InfiniteTalk Logic for Audio-Driven Video")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to Wan2.2-S2V-14B checkpoint")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (WAV)")
    parser.add_argument("--input_json", type=str, default=None, help="Path to input JSON (single or multi-person)")
    parser.add_argument("--input_video", type=str, default=None, help="Path to input video for V2V mode")
    parser.add_argument("--size", type=str, default="1024*704", help="Output resolution (e.g., '1024*704')")
    parser.add_argument("--sample_steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--mode", type=str, default="streaming", choices=["streaming", "clip"], help="Generation mode")
    parser.add_argument("--motion_frame", type=int, default=9, help="Frames for motion continuity")
    parser.add_argument("--max_frame_num", type=int, default=1000, help="Max frames for video")
    parser.add_argument("--save_file", type=str, default="output.mp4", help="Output video path")
    parser.add_argument("--offload_model", type=bool, default=True, help="Offload model to CPU")
    parser.add_argument("--quant", type=str, default="fp8", choices=["none", "fp8"], help="Quantization mode")
    parser.add_argument("--sample_audio_guide_scale", type=float, default=2.0, help="Audio guidance scale")
    parser.add_argument("--use_teacache", type=bool, default=False, help="Simulate TeaCache acceleration")
    parser.add_argument("--lora_dir", type=str, default=None, help="Path to LoRA weights (e.g., FusionX)")
    parser.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scale")
    return parser.parse_args()

def main():
    """
    Main function to generate audio-driven video using Wan2.2-S2V-14B with InfiniteTalk logic.
    """
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    try:
        # Initialize Wan2.2 environment
        initializer = Wan22Initializer(args.ckpt_dir, task="s2v-14b")
        logging.info("Wan2.2 environment initialized")

        # Load configuration
        config = Wan22Config(resolution=args.size).get_config()
        logging.info(f"Configuration loaded: {config}")

        # Initialize model
        model = Wan22Model(
            ckpt_dir=args.ckpt_dir,
            task=config.task,
            device=device,
            quant=args.quant,
            lora_dir=args.lora_dir,
            lora_scale=args.lora_scale
        )
        logging.info("Wan2.2 model initialized")

        # Initialize pipeline
        pipeline = Wan22Pipeline(model, offload_model=args.offload_model)
        logging.info("Wan2.2 pipeline initialized")

        # Load inputs (JSON for I2V, video for V2V)
        if args.input_video:
            logging.info(f"Loading video input: {args.input_video}")
            image_paths = [Wan22Utils.extract_first_frame(args.input_video)]
        elif args.input_json:
            logging.info(f"Loading JSON input: {args.input_json}")
            with open(args.input_json, 'r') as f:
                image_paths = json.load(f).get('images', [])
        else:
            logging.error("No input JSON or video provided")
            raise ValueError("Either --input_json or --input_video must be provided")

        if not image_paths:
            logging.error("No input images found")
            raise ValueError("No input images provided")

        # Generate videos for each person
        videos = []
        temp_files = []
        for idx, image_path in enumerate(image_paths):
            try:
                logging.info(f"Processing image {idx+1}/{len(image_paths)}: {image_path}")
                preprocessed_image = Wan22Utils.preprocess_image(image_path, config.get_resolution())
                video = pipeline.generate(
                    image_path=preprocessed_image,
                    audio_path=args.audio,
                    num_steps=args.sample_steps,
                    mode=args.mode,
                    motion_frame=args.motion_frame,
                    max_frame_num=args.max_frame_num,
                    guide_scale=args.sample_audio_guide_scale
                )
                temp_file = f"temp_video_{idx}.mp4"
                pipeline.save_video(video, temp_file, fps=24)
                videos.append(temp_file)
                temp_files.append(temp_file)
                logging.info(f"Generated temporary video: {temp_file}")
            except Exception as e:
                logging.error(f"Error generating video for image {image_path}: {str(e)}")
                raise

        # Combine videos for multi-person
        try:
            logging.info(f"Combining {len(videos)} videos into {args.save_file}")
            pipeline.combine_videos(videos, args.save_file)
            logging.info(f"Final video saved to {args.save_file}")
        except Exception as e:
            logging.error(f"Error combining videos: {str(e)}")
            raise

        # Clean up temporary files
        try:
            Wan22Utils.clean_temp_files(temp_files)
            logging.info("Temporary files cleaned up")
        except Exception as e:
            logging.error(f"Error cleaning temporary files: {str(e)}")
            raise

    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
        raise
    except ValueError as e:
        logging.error(f"Configuration or input error: {str(e)}")
        raise
    except RuntimeError as e:
        logging.error(f"Runtime error during generation: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    main()