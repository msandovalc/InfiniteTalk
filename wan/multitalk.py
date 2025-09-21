
# File: F:\Development\PyCharm\Projects\InfiniteTalk\wan\multitalk.py
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import gc
import logging
import json
import math
import importlib
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial
from PIL import Image
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from diffusers.models.modeling_utils import no_init_weights, ContextManagers
import accelerate
from .distributed.fsdp import shard_model
from .modules.clip import CLIPModel
from .modules.multitalk_model import WanModel, WanLayerNorm, WanRMSNorm
from .modules.t5 import T5EncoderModel, T5LayerNorm, T5RelativeEmbedding
from .modules.vae import WanVAE, CausalConv3d, RMS_norm, Upsample
from .utils.multitalk_utils import MomentumBuffer, adaptive_projected_guidance, match_and_blend_colors
from src.vram_management import AutoWrappedQLinear, AutoWrappedLinear, AutoWrappedModule, enable_vram_management
from wan.utils.utils import convert_video_to_h264, extract_specific_frames, get_video_codec
from wan.wan_lora import WanLoraWrapper
from safetensors.torch import load_file
from optimum.quanto import quantize, freeze, qint8, requantize
import optimum.quanto.nn.qlinear as qlinear

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def torch_gc():
    """Clear CUDA memory."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logging.info("CUDA memory cleared")
    except Exception as e:
        logging.error(f"Error clearing CUDA memory: {str(e)}")
        raise


def to_param_dtype_fp32only(model, param_dtype):
    """Convert model parameters to specified dtype, excluding quantized tensors."""
    try:
        for module in model.modules():
            for name, param in module.named_parameters(recurse=False):
                if param.dtype == torch.float32 and param.__class__.__name__ not in ['WeightQBytesTensor']:
                    param.data = param.data.to(param_dtype)
            for name, buf in module.named_buffers(recurse=False):
                if buf.dtype == torch.float32 and buf.__class__.__name__ not in ['WeightQBytesTensor']:
                    module._buffers[name] = buf.to(param_dtype)
        logging.info(f"Converted model parameters to dtype: {param_dtype}")
    except Exception as e:
        logging.error(f"Error converting parameters to dtype: {str(e)}")
        raise


def resize_and_centercrop(cond_image, target_size):
    """
    Resize image or tensor to the target size without padding.
    """
    try:
        if isinstance(cond_image, torch.Tensor):
            _, orig_h, orig_w = cond_image.shape
        else:
            orig_h, orig_w = cond_image.height, cond_image.width

        target_h, target_w = target_size
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        scale = max(scale_h, scale_w)
        final_h = math.ceil(scale * orig_h)
        final_w = math.ceil(scale * orig_w)

        if isinstance(cond_image, torch.Tensor):
            if len(cond_image.shape) == 3:
                cond_image = cond_image[None]
            resized_tensor = nn.functional.interpolate(cond_image, size=(final_h, final_w), mode='nearest').contiguous()
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
            cropped_tensor = cropped_tensor.squeeze(0)
        else:
            resized_image = cond_image.resize((final_w, final_h), resample=Image.BILINEAR)
            resized_image = np.array(resized_image)
            resized_tensor = torch.from_numpy(resized_image)[None, ...].permute(0, 3, 1, 2).contiguous()
            cropped_tensor = transforms.functional.center_crop(resized_tensor, target_size)
            cropped_tensor = cropped_tensor[:, :, None, :, :]

        logging.info(f"Image resized and cropped to {target_size}")
        return cropped_tensor
    except Exception as e:
        logging.error(f"Error resizing image: {str(e)}")
        raise


def timestep_transform(t, shift=5.0, num_timesteps=1000):
    """Transform timesteps for diffusion process."""
    try:
        t = t / num_timesteps
        new_t = shift * t / (1 + (shift - 1) * t)
        new_t = new_t * num_timesteps
        return new_t
    except Exception as e:
        logging.error(f"Error transforming timestep: {str(e)}")
        raise


class InfiniteTalkPipeline:
    """
    Pipeline for InfiniteTalk, adapted for Wan models with VRAM management.
    """

    def __init__(
            self,
            config,
            checkpoint_dir,
            quant_dir=None,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            init_on_cpu=True,
            num_timesteps=1000,
            use_timestep_transform=True,
            lora_dir=None,
            lora_scales=None,
            quant=None,
            dit_path=None,
            infinitetalk_dir=None,
    ):
        """
        Initializes the image-to-video generation model components.
        """
        if quant is not None and quant not in ("int8", "fp8"):
            logging.error("Quant must be 'int8', 'fp8', or None")
            raise ValueError("quant must be 'int8', 'fp8', or None(default fp32 model)")

        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        shard_fn = partial(shard_model, device_id=device_id)

        try:
            # Initialize text encoder
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
                shard_fn=shard_fn if t5_fsdp else None,
                quant=quant,
                quant_dir=os.path.dirname(quant_dir) if quant_dir is not None else None,
            )
            logging.info("Text encoder initialized")

            # Initialize VAE
            self.vae_stride = config.vae_stride
            self.patch_size = config.patch_size
            self.vae = WanVAE(
                vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
                device=self.device
            )
            logging.info("VAE initialized")

            # Initialize CLIP
            self.clip = CLIPModel(
                dtype=config.clip_dtype,
                device=self.device,
                checkpoint_path=os.path.join(checkpoint_dir, config.clip_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir, config.clip_tokenizer)
            )
            logging.info("CLIP model initialized")

            # Initialize WanModel
            logging.info(f"Creating WanModel from {checkpoint_dir}")
            if quant is not None:
                logging.info(f"Loading Quantized MultiTalk from {quant_dir}")
                with torch.device('meta'):
                    wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                    self.model = WanModel(weight_init=False, **wan_config)
                    torch_gc()
                model_state_dict = load_file(quant_dir)
                map_json_path = os.path.join(quant_dir.replace('safetensors', 'json'))
                self.model.init_freqs()
                with open(map_json_path, "r") as f:
                    quantization_map = json.load(f)
                requantize(self.model, model_state_dict, quantization_map, device='cpu')
            else:
                init_contexts = [no_init_weights()]
                init_contexts.append(accelerate.init_empty_weights())
                wan_config = json.load(open(os.path.join(checkpoint_dir, "config.json")))
                self.model = WanModel(weight_init=False, **wan_config).to(dtype=self.param_dtype)
                weight_files = [
                                   f"{checkpoint_dir}/diffusion_pytorch_model-0000{i}-of-00007.safetensors" for i in
                                   range(1, 8)
                               ] + [f"{infinitetalk_dir}"]
                merged_state_dict = {}
                for weight_file in weight_files:
                    sd = load_file(weight_file)
                    merged_state_dict.update(sd)
                self.model.load_state_dict(merged_state_dict)

            self.model.eval().requires_grad_(False)
            to_param_dtype_fp32only(self.model, self.param_dtype)
            logging.info("WanModel initialized")

            # Apply LoRA if provided
            if lora_dir is not None and quant is None:
                lora_wrapper = WanLoraWrapper(self.model)
                for lora_path, lora_scale in zip(lora_dir, lora_scales):
                    lora_name = lora_wrapper.load_lora(lora_path)
                    lora_wrapper.apply_lora(lora_name, lora_scale, param_dtype=self.param_dtype, device=self.device)
                logging.info("LoRA weights applied")

            # Distributed setup
            if t5_fsdp or dit_fsdp or use_usp:
                init_on_cpu = False
            if use_usp:
                from xfuser.core.distributed import get_sequence_parallel_world_size
                from .distributed.xdit_context_parallel import (
                    usp_dit_forward_multitalk,
                    usp_attn_forward_multitalk,
                    usp_crossattn_multi_forward_multitalk
                )
                for block in self.model.blocks:
                    block.self_attn.forward = types.MethodType(usp_attn_forward_multitalk, block.self_attn)
                    block.audio_cross_attn.forward = types.MethodType(usp_crossattn_multi_forward_multitalk,
                                                                      block.audio_cross_attn)
                self.model.forward = types.MethodType(usp_dit_forward_multitalk, self.model)
                self.sp_size = get_sequence_parallel_world_size()
                logging.info("USP distributed setup completed")
            else:
                self.sp_size = 1

            if dist.is_initialized():
                dist.barrier()
            if dit_fsdp:
                self.model = shard_fn(self.model)
            else:
                if not init_on_cpu:
                    self.model.to(self.device)
            logging.info("Model distribution setup completed")

            self.sample_neg_prompt = config.sample_neg_prompt
            self.num_timesteps = num_timesteps
            self.use_timestep_transform = use_timestep_transform
            self.cpu_offload = False
            self.model_names = ["model"]
            self.vram_management = False
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise

    def add_noise(self, original_samples: torch.FloatTensor, noise: torch.FloatTensor,
                  timesteps: torch.IntTensor) -> torch.FloatTensor:
        """
        Compatible with diffusers add_noise().
        """
        try:
            timesteps = timesteps.float() / self.num_timesteps
            timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape) - 1))
            result = (1 - timesteps) * original_samples + timesteps * noise
            logging.info("Noise added to samples")
            return result
        except Exception as e:
            logging.error(f"Error adding noise: {str(e)}")
            raise

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        """Enable VRAM management for model."""
        try:
            dtype = next(iter(self.model.parameters())).dtype
            enable_vram_management(
                self.model,
                module_map={
                    qlinear.QLinear: AutoWrappedQLinear,
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    WanLayerNorm: AutoWrappedModule,
                    WanRMSNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.param_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.param_dtype,
                    computation_device=self.device,
                ),
            )
            self.enable_cpu_offload()
            logging.info("VRAM management enabled")
        except Exception as e:
            logging.error(f"Error enabling VRAM management: {str(e)}")
            raise

    def enable_cpu_offload(self):
        """Enable CPU offloading."""
        try:
            self.cpu_offload = True
            logging.info("CPU offload enabled")
        except Exception as e:
            logging.error(f"Error enabling CPU offload: {str(e)}")
            raise

    def load_models_to_device(self, loadmodel_names=[]):
        """Load specified models to device, offload others if CPU offload is enabled."""
        try:
            if not self.cpu_offload:
                return
            for model_name in self.model_names:
                if model_name not in loadmodel_names:
                    model = getattr(self, model_name)
                    if not isinstance(model, nn.Module):
                        model = model.model
                    if model is not None:
                        if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                            for module in model.modules():
                                if hasattr(module, "offload"):
                                    module.offload()
                        else:
                            model.cpu()
            for model_name in loadmodel_names:
                model = getattr(self, model_name)
                if not isinstance(model, nn.Module):
                    model = model.model
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        for module in model.modules():
                            if hasattr(module, "onload"):
                                module.onload()
                    else:
                        model.to(self.device)
            torch.cuda.empty_cache()
            logging.info("Models loaded to device")
        except Exception as e:
            logging.error(f"Error loading models to device: {str(e)}")
            raise

    def generate_infinitetalk(
            self,
            input_data,
            size_buckget='infinitetalk-480',
            motion_frame=25,
            frame_num=81,
            shift=5.0,
            sampling_steps=40,
            text_guide_scale=5.0,
            audio_guide_scale=4.0,
            n_prompt="",
            seed=-1,
            offload_model=True,
            max_frames_num=1000,
            face_scale=0.05,
            progress=True,
            color_correction_strength=0.0,
            extra_args=None
    ):
        """
        Generates video frames from input image and text prompt using diffusion process.
        """
        try:
            # Initialize TeaCache
            if extra_args.use_teacache:
                self.model.teacache_init(
                    sample_steps=sampling_steps,
                    teacache_thresh=extra_args.teacache_thresh,
                    model_scale=extra_args.size,
                )
            else:
                self.model.disable_teacache()
            logging.info("TeaCache configured")

            # Process input video
            input_prompt = input_data['prompt']
            cond_file_path = input_data['cond_video']
            codec = get_video_codec(cond_file_path)
            if codec == 'av1':
                output_video_path = 'tmp/_input_h264.mp4'
                logging.info(f"Converting {cond_file_path} from AV1 to H.264")
                convert_video_to_h264(cond_file_path, output_video_path)
                cond_file_path = output_video_path
                logging.info(f"Conversion complete: {output_video_path}")
            else:
                logging.info("No video conversion needed")

            # Extract and preprocess conditioning image
            cond_image = extract_specific_frames(cond_file_path, 0)
            bucket_config_module = importlib.import_module("wan.utils.multitalk_utils")
            if size_buckget == 'infinitetalk-480':
                bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_627')
            elif size_buckget == 'infinitetalk-720':
                bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_960')

            src_h, src_w = cond_image.height, cond_image.width
            ratio = src_h / src_w
            closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x) - ratio))[0]
            target_h, target_w = bucket_config[closest_bucket][0]
            cond_image = resize_and_centercrop(cond_image, (target_h, target_w))
            cond_image = cond_image / 255
            cond_image = (cond_image - 0.5) * 2  # Normalization
            cond_image = cond_image.to(self.device)
            logging.info(f"Conditioning image processed: {target_h}x{target_w}")

            # Store original image for color correction
            original_color_reference = None
            if color_correction_strength > 0.0:
                original_color_reference = cond_image.clone()
                logging.info("Stored original image for color correction")

            # Read audio embeddings
            audio_embedding_path_1 = input_data['cond_audio']['person1']
            HUMAN_NUMBER = 1 if len(input_data['cond_audio']) == 1 else 2
            audio_embedding_path_2 = input_data['cond_audio'].get('person2')
            full_audio_embs = []
            audio_embedding_paths = [audio_embedding_path_1, audio_embedding_path_2]
            for human_idx in range(HUMAN_NUMBER):
                audio_embedding_path = audio_embedding_paths[human_idx]
                if not os.path.exists(audio_embedding_path):
                    logging.error(f"Audio embedding file not found: {audio_embedding_path}")
                    continue
                full_audio_emb = torch.load(audio_embedding_path)
                if torch.isnan(full_audio_emb).any():
                    logging.error(f"Invalid audio embedding: {audio_embedding_path}")
                    continue
                if full_audio_emb.shape[0] <= frame_num:
                    logging.error(f"Audio embedding too short: {audio_embedding_path}")
                    continue
                full_audio_embs.append(full_audio_emb)
            if len(full_audio_embs) != HUMAN_NUMBER:
                logging.error("Audio file error: length or existence issue")
                raise ValueError("Audio file not exists or length not satisfies frame nums")
            logging.info(f"Loaded {HUMAN_NUMBER} audio embeddings")

            # Preprocess text embedding
            if n_prompt == "":
                n_prompt = self.sample_neg_prompt
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context, context_null = self.text_encoder([input_prompt, n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
            logging.info("Text embeddings processed")
            torch_gc()

            # Prepare params for video generation
            indices = (torch.arange(2 * 2 + 1) - 2) * 1
            clip_length = frame_num
            is_first_clip = True
            arrive_last_frame = False
            cur_motion_frames_num = 1
            audio_start_idx = 0
            audio_end_idx = audio_start_idx + clip_length
            gen_video_list = []
            torch_gc()

            # Set random seed
            seed = seed if seed >= 0 else random.randint(0, 99999999)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True
            logging.info(f"Random seed set: {seed}")

            # Start video generation iteratively
            while True:
                audio_embs = []
                for human_idx in range(HUMAN_NUMBER):
                    center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
                    center_indices = torch.clamp(center_indices, min=0, max=full_audio_embs[human_idx].shape[0] - 1)
                    audio_emb = full_audio_embs[human_idx][center_indices][None, ...].to(self.device)
                    audio_embs.append(audio_emb)
                audio_embs = torch.concat(audio_embs, dim=0).to(self.param_dtype)
                logging.info("Audio embeddings processed for current clip")
                torch_gc()

                h, w = cond_image.shape[-2], cond_image.shape[-1]
                lat_h, lat_w = h // self.vae_stride[1], w // self.vae_stride[2]
                max_seq_len = ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (
                            self.patch_size[1] * self.patch_size[2])
                max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

                noise = torch.randn(16, (frame_num - 1) // 4 + 1, lat_h, lat_w, dtype=torch.float32, device=self.device)
                logging.info("Noise initialized")

                # Get mask
                msk = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
                msk[:, 1:] = 0
                msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
                msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                msk = msk.transpose(1, 2).to(self.param_dtype)
                logging.info("Mask generated")

                with torch.no_grad():
                    self.clip.model.to(self.device)
                    clip_context = self.clip.visual(cond_image[:, :, -1:, :, :]).to(self.param_dtype)
                    if offload_model:
                        self.clip.model.cpu()
                    torch_gc()
                    logging.info("CLIP embedding generated")

                    video_frames = torch.zeros(1, cond_image.shape[1], frame_num - cond_image.shape[2], target_h,
                                               target_w).to(self.device)
                    padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
                    y = self.vae.encode(padding_frames_pixels_values)
                    y = torch.stack(y).to(self.param_dtype)
                    cur_motion_frames_latent_num = int(1 + (cur_motion_frames_num - 1) // 4)

                    if is_first_clip:
                        latent_motion_frames = self.vae.encode(cond_image)[0]
                    else:
                        latent_motion_frames = self.vae.encode(cond_frame)[0]

                    y = torch.concat([msk, y], dim=1)
                    torch_gc()
                    logging.info("VAE encoding completed")

                # Construct human mask
                human_masks = []
                if HUMAN_NUMBER == 1:
                    background_mask = torch.ones([src_h, src_w])
                    human_mask1 = torch.ones([src_h, src_w])
                    human_mask2 = torch.ones([src_h, src_w])
                    human_masks = [human_mask1, human_mask2, background_mask]
                elif HUMAN_NUMBER == 2:
                    if 'bbox' in input_data:
                        assert len(input_data['bbox']) == len(
                            input_data['cond_audio']), "The number of target bbox should match cond_audio"
                        background_mask = torch.zeros([src_h, src_w])
                        for _, person_bbox in input_data['bbox'].items():
                            x_min, y_min, x_max, y_max = person_bbox
                            human_mask = torch.zeros([src_h, src_w])
                            human_mask[int(x_min):int(x_max), int(y_min):int(y_max)] = 1
                            background_mask += human_mask
                            human_masks.append(human_mask)
                    else:
                        x_min, x_max = int(src_h * face_scale), int(src_h * (1 - face_scale))
                        background_mask = torch.zeros([src_h, src_w])
                        human_mask1 = torch.zeros([src_h, src_w])
                        human_mask2 = torch.zeros([src_h, src_w])
                        lefty_min, lefty_max = int((src_w // 2) * face_scale), int((src_w // 2) * (1 - face_scale))
                        righty_min, righty_max = int((src_w // 2) * face_scale + (src_w // 2)), int(
                            (src_w // 2) * (1 - face_scale) + (src_w // 2))
                        human_mask1[x_min:x_max, lefty_min:lefty_max] = 1
                        human_mask2[x_min:x_max, righty_min:righty_max] = 1
                        background_mask += human_mask1
                        background_mask += human_mask2
                        human_masks = [human_mask1, human_mask2]
                    background_mask = torch.where(background_mask > 0, torch.tensor(0), torch.tensor(1))
                    human_masks.append(background_mask)

                ref_target_masks = torch.stack(human_masks, dim=0).to(self.device)
                ref_target_masks = resize_and_centercrop(ref_target_masks, (target_h, target_w))
                ref_target_masks = F.interpolate(ref_target_masks.unsqueeze(0), size=(lat_h, lat_w),
                                                 mode='nearest').squeeze()
                ref_target_masks = (ref_target_masks > 0).float().to(self.device)
                logging.info("Human masks constructed")

                @contextmanager
                def noop_no_sync():
                    yield

                no_sync = getattr(self.model, 'no_sync', noop_no_sync)

                with torch.no_grad(), no_sync():
                    timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
                    timesteps.append(0.)
                    timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
                    if self.use_timestep_transform:
                        timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps) for t in
                                     timesteps]
                    logging.info("Timesteps prepared")

                    latent = noise
                    arg_c = {
                        'context': [context],
                        'clip_fea': clip_context,
                        'seq_len': max_seq_len,
                        'y': y,
                        'audio': audio_embs,
                        'ref_target_masks': ref_target_masks
                    }
                    arg_null_text = {
                        'context': [context_null],
                        'clip_fea': clip_context,
                        'seq_len': max_seq_len,
                        'y': y,
                        'audio': audio_embs,
                        'ref_target_masks': ref_target_masks
                    }
                    arg_null_audio = {
                        'context': [context],
                        'clip_fea': clip_context,
                        'seq_len': max_seq_len,
                        'y': y,
                        'audio': torch.zeros_like(audio_embs)[-1:],
                        'ref_target_masks': ref_target_masks
                    }
                    arg_null = {
                        'context': [context_null],
                        'clip_fea': clip_context,
                        'seq_len': max_seq_len,
                        'y': y,
                        'audio': torch.zeros_like(audio_embs)[-1:],
                        'ref_target_masks': ref_target_masks
                    }
                    logging.info("Conditioning arguments prepared")

                    if not self.vram_management:
                        self.model.to(self.device)
                    else:
                        self.load_models_to_device(["model"])
                    logging.info("Model loaded for inference")

                    if not is_first_clip:
                        latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                        motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                        add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[0])
                        _, T_m, _, _ = add_latent.shape
                        latent[:, :T_m] = add_latent

                    if extra_args.use_apg:
                        text_momentumbuffer = MomentumBuffer(extra_args.apg_momentum)
                        audio_momentumbuffer = MomentumBuffer(extra_args.apg_momentum)

                    progress_wrap = partial(tqdm, total=len(timesteps) - 1) if progress else (lambda x: x)
                    for i in progress_wrap(range(len(timesteps) - 1)):
                        timestep = timesteps[i]
                        latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
                        latent_model_input = [latent.to(self.device)]

                        noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                        torch_gc()

                        if math.isclose(text_guide_scale, 1.0):
                            noise_pred_drop_audio = self.model(latent_model_input, t=timestep, **arg_null_audio)[0]
                            torch_gc()
                        else:
                            noise_pred_drop_text = self.model(latent_model_input, t=timestep, **arg_null_text)[0]
                            torch_gc()
                            noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                            torch_gc()

                        if extra_args.use_apg:
                            if math.isclose(text_guide_scale, 1.0):
                                diff_uncond_audio = noise_pred_cond - noise_pred_drop_audio
                                noise_pred = noise_pred_cond + (audio_guide_scale - 1) * adaptive_projected_guidance(
                                    diff_uncond_audio, noise_pred_cond, momentum_buffer=audio_momentumbuffer,
                                    norm_threshold=extra_args.apg_norm_threshold)
                            else:
                                diff_uncond_text = noise_pred_cond - noise_pred_drop_text
                                diff_uncond_audio = noise_pred_drop_text - noise_pred_uncond
                                noise_pred = noise_pred_cond + (text_guide_scale - 1) * adaptive_projected_guidance(
                                    diff_uncond_text, noise_pred_cond, momentum_buffer=text_momentumbuffer,
                                    norm_threshold=extra_args.apg_norm_threshold) + (
                                                         audio_guide_scale - 1) * adaptive_projected_guidance(
                                    diff_uncond_audio, noise_pred_cond, momentum_buffer=audio_momentumbuffer,
                                    norm_threshold=extra_args.apg_norm_threshold)
                        else:
                            if math.isclose(text_guide_scale, 1.0):
                                noise_pred = noise_pred_drop_audio + audio_guide_scale * (
                                            noise_pred_cond - noise_pred_drop_audio)
                            else:
                                noise_pred = noise_pred_uncond + text_guide_scale * (
                                            noise_pred_cond - noise_pred_drop_text) + \
                                             audio_guide_scale * (noise_pred_drop_text - noise_pred_uncond)
                        noise_pred = -noise_pred

                        dt = timesteps[i] - timesteps[i + 1]
                        dt = dt / self.num_timesteps
                        latent = latent + noise_pred * dt[:, None, None, None]

                        if not is_first_clip:
                            latent_motion_frames = latent_motion_frames.to(latent.dtype).to(self.device)
                            motion_add_noise = torch.randn_like(latent_motion_frames).contiguous()
                            add_latent = self.add_noise(latent_motion_frames, motion_add_noise, timesteps[i + 1])
                            _, T_m, _, _ = add_latent.shape
                            latent[:, :T_m] = add_latent

                        latent[:, :cur_motion_frames_latent_num] = latent_motion_frames
                        x0 = [latent.to(self.device)]
                        del latent_model_input, timestep

                    if offload_model:
                        if not self.vram_management:
                            self.model.cpu()
                    torch_gc()
                    logging.info("Inference completed")

                    videos = self.vae.decode(x0)
                    videos = torch.stack(videos).cpu()
                    if color_correction_strength > 0.0 and original_color_reference is not None:
                        videos = match_and_blend_colors(videos, original_color_reference, color_correction_strength)
                    logging.info("Videos decoded and color-corrected")

                    if is_first_clip:
                        gen_video_list.append(videos)
                    else:
                        gen_video_list.append(videos[:, :, cur_motion_frames_num:])

                    if arrive_last_frame:
                        break

                    is_first_clip = False
                    cur_motion_frames_num = motion_frame
                    cond_frame = videos[:, :, -cur_motion_frames_num:].to(torch.float32).to(self.device)
                    audio_start_idx += (frame_num - cur_motion_frames_num)
                    audio_end_idx = audio_start_idx + clip_length

                    cond_image = extract_specific_frames(cond_file_path, audio_start_idx)
                    cond_image = resize_and_centercrop(cond_image, (target_h, target_w))
                    cond_image = cond_image / 255
                    cond_image = (cond_image - 0.5) * 2
                    cond_image = cond_image.to(self.device)

                    if audio_end_idx >= min(max_frames_num, len(full_audio_embs[0])):
                        arrive_last_frame = True
                        miss_lengths = []
                        source_frames = []
                        for human_inx in range(HUMAN_NUMBER):
                            source_frame = len(full_audio_embs[human_inx])
                            source_frames.append(source_frame)
                            if audio_end_idx >= len(full_audio_embs[human_inx]):
                                miss_length = audio_end_idx - len(full_audio_embs[human_inx]) + 3
                                add_audio_emb = torch.flip(full_audio_embs[human_inx][-1 * miss_length:], dims=[0])
                                full_audio_embs[human_inx] = torch.cat([full_audio_embs[human_inx], add_audio_emb],
                                                                       dim=0)
                                miss_lengths.append(miss_length)
                            else:
                                miss_lengths.append(0)

                    if max_frames_num <= frame_num:
                        break

                    torch_gc()
                    if offload_model:
                        torch.cuda.synchronize()
                    if dist.is_initialized():
                        dist.barrier()

                gen_video_samples = torch.cat(gen_video_list, dim=2)[:, :, :int(max_frames_num)]
                gen_video_samples = gen_video_samples.to(torch.float32)
                if max_frames_num > frame_num and sum(miss_lengths) > 0:
                    gen_video_samples = gen_video_samples[:, :, :full_audio_emb.shape[0]]

                if dist.is_initialized():
                    dist.barrier()

                del noise, latent
                torch_gc()
                logging.info("Video generation completed")

                return gen_video_samples[0] if self.rank == 0 else None
        except Exception as e:
            logging.error(f"Error in generate_infinitetalk: {str(e)}")
            raise
