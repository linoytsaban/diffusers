# Copyright 2025 The Framepack Team, The HunyuanVideo Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    LlamaModel,
    LlamaTokenizerFast,
    SiglipImageProcessor,
    SiglipVisionModel,
)

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...image_processor import PipelineImageInput
from ...loaders import HunyuanVideoLoraLoaderMixin
from ...models import AutoencoderKLHunyuanVideo, HunyuanVideoFramepackTransformer3DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import HunyuanVideoFramepackPipelineOutput

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)

EXAMPLE_DOC_STRING = """
    Examples:
        ##### Image-to-Video & Image-to-Image

        ```python
        >>> import torch
        >>> from diffusers import HunyuanVideoFramepackPipeline, HunyuanVideoFramepackTransformer3DModel
        >>> from diffusers.utils import export_to_video, load_image, export_to_pil
        >>> from transformers import SiglipImageProcessor, SiglipVisionModel

        >>> transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
        ...     "lllyasviel/FramePackI2V_HY", torch_dtype=torch.bfloat16
        ... )
        >>> feature_extractor = SiglipImageProcessor.from_pretrained(
        ...     "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
        ... )
        >>> image_encoder = SiglipVisionModel.from_pretrained(
        ...     "lllyasviel/flux_redux_bfl", subfolder="image_encoder", torch_dtype=torch.float16
        ... )
        >>> pipe = HunyuanVideoFramepackPipeline.from_pretrained(
        ...     "hunyuanvideo-community/HunyuanVideo",
        ...     transformer=transformer,
        ...     feature_extractor=feature_extractor,
        ...     image_encoder=image_encoder,
        ...     torch_dtype=torch.float16,
        ... )
        >>> pipe.vae.enable_tiling()
        >>> pipe.to("cuda")

        >>> image_input = load_image(
        ...     "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/penguin.png"
        ... )

        >>> # Video output
        >>> output_video_frames = pipe(
        ...     image=image_input,
        ...     prompt="A penguin dancing in the snow",
        ...     height=832,
        ...     width=480,
        ...     num_frames=91, 
        ...     num_inference_steps=30,
        ...     guidance_scale=9.0,
        ...     generator=torch.Generator().manual_seed(0),
        ... ).frames[0]
        >>> export_to_video(output_video_frames, "output_video.mp4", fps=30)

        >>> # Single frame (image) output
        >>> output_single_image = pipe(
        ...     image=image_input,
        ...     prompt="A penguin dancing in the snow, photorealistic",
        ...     height=512, 
        ...     width=512,
        ...     num_frames=1, # Key for single frame
        ...     num_inference_steps=30,
        ...     guidance_scale=9.0,
        ...     generator=torch.Generator().manual_seed(1),
        ...     output_type="pil",
        ...     # Optionally add single frame options from the original script
        ...     # one_frame_inference_options=["no_post", "no_2x", "no_4x"] 
        ... ).frames[0][0] 
        >>> output_single_image.save("output_image.png")
        ```
"""

DEFAULT_PROMPT_TEMPLATE = {
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "4. background environment, light, style and atmosphere."
        "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}


def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class HunyuanVideoFramepackPipeline(DiffusionPipeline, HunyuanVideoLoraLoaderMixin):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->image_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
            self,
            text_encoder: LlamaModel,
            tokenizer: LlamaTokenizerFast,
            transformer: HunyuanVideoFramepackTransformer3DModel,
            vae: AutoencoderKLHunyuanVideo,
            scheduler: FlowMatchEulerDiscreteScheduler,
            text_encoder_2: CLIPTextModel,
            tokenizer_2: CLIPTokenizer,
            image_encoder: SiglipVisionModel,
            feature_extractor: SiglipImageProcessor,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )

        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_llama_prompt_embeds(
            self,
            prompt: Union[str, List[str]],
            prompt_template: Dict[str, Any],
            num_videos_per_prompt: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            max_sequence_length: int = 256,
            num_hidden_layers_to_skip: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        prompt = [prompt_template["template"].format(p) for p in prompt]
        crop_start = prompt_template.get("crop_start", None)
        if crop_start is None:
            prompt_template_input = self.tokenizer(
                prompt_template["template"],
                padding="max_length",
                return_tensors="pt",
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=False,
            )
            crop_start = prompt_template_input["input_ids"].shape[-1]
            crop_start -= 2
        max_sequence_length += crop_start
        text_inputs = self.tokenizer(
            prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_length=False,
            return_overflowing_tokens=False,
            return_attention_mask=True,
        )
        text_input_ids = text_inputs.input_ids.to(device=device)
        prompt_attention_mask = text_inputs.attention_mask.to(device=device)
        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_attention_mask,
            output_hidden_states=True,
        ).hidden_states[-(num_hidden_layers_to_skip + 1)]
        prompt_embeds = prompt_embeds.to(dtype=dtype)
        if crop_start is not None and crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_attention_mask = prompt_attention_mask[:, crop_start:]
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(1, num_videos_per_prompt)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)
        return prompt_embeds, prompt_attention_mask

    def _get_clip_prompt_embeds(
            self,
            prompt: Union[str, List[str]],
            num_videos_per_prompt: int = 1,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            max_sequence_length: int = 77,
    ) -> torch.Tensor:
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_2.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, max_sequence_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {max_sequence_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False).pooler_output
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, -1)
        return prompt_embeds

    def encode_prompt(
            self,
            prompt: Union[str, List[str]],
            prompt_2: Union[str, List[str]] = None,
            prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
            num_videos_per_prompt: int = 1,
            prompt_embeds: Optional[torch.Tensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            prompt_attention_mask: Optional[torch.Tensor] = None,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            max_sequence_length: int = 256,
    ):
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_llama_prompt_embeds(
                prompt,
                prompt_template,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=max_sequence_length,
            )
        if pooled_prompt_embeds is None:
            if prompt_2 is None:
                prompt_2 = prompt
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt_2,
                num_videos_per_prompt,
                device=device,
                dtype=dtype,
                max_sequence_length=77,
            )
        return prompt_embeds, pooled_prompt_embeds, prompt_attention_mask

    def encode_image(
            self, image: torch.Tensor, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ):
        device = device or self._execution_device
        image = (image + 1) / 2.0
        image = self.feature_extractor(images=image, return_tensors="pt", do_rescale=False).to(
            device=device, dtype=self.image_encoder.dtype
        )
        image_embeds = self.image_encoder(**image).last_hidden_state
        return image_embeds.to(dtype=dtype)

    def check_inputs(
            self,
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=None,
            callback_on_step_end_tensor_inputs=None,
            prompt_template=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")
        if callback_on_step_end_tensor_inputs is not None and not all(
                k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")
        if prompt_template is not None:
            if not isinstance(prompt_template, dict):
                raise ValueError(f"`prompt_template` has to be of type `dict` but is {type(prompt_template)}")
            if "template" not in prompt_template:
                raise ValueError(
                    f"`prompt_template` has to contain a key `template` but only found {prompt_template.keys()}"
                )

    def prepare_latents(
            self,
            batch_size: int = 1,
            num_channels_latents: int = 16,
            height: int = 720,
            width: int = 1280,
            num_frames: int = 129,  # For video, this is num_latent_frames_in_section; for image, it's 1
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
            # is_one_frame_inference: bool = False, # Not strictly needed if num_frames passed correctly
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        # num_frames here is the number of latent frames for the noise tensor for ONE DiT step
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,  # Use directly, will be 1 for single image, or (latent_window_size-1) for video section
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def prepare_image_latents(
            self,
            image: torch.Tensor,
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = device or self._execution_device
        if latents is None:
            image = image.unsqueeze(2).to(device=device, dtype=self.vae.dtype)
            latents = self.vae.encode(image).latent_dist.sample(generator=generator)
            latents = latents * self.vae.config.scaling_factor
        return latents.to(device=device, dtype=dtype)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        self.vae.disable_tiling()

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            image: PipelineImageInput,
            last_image: Optional[PipelineImageInput] = None,
            prompt: Union[str, List[str]] = None,
            prompt_2: Union[str, List[str]] = None,
            negative_prompt: Union[str, List[str]] = None,
            negative_prompt_2: Union[str, List[str]] = None,
            height: int = 720,
            width: int = 1280,
            num_frames: int = 129,  # TARGET pixel frames for video, or 1 for image
            latent_window_size: int = 9,
            num_inference_steps: int = 50,
            sigmas: List[float] = None,
            true_cfg_scale: float = 1.0,
            guidance_scale: float = 6.0,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            image_latents: Optional[torch.Tensor] = None,
            last_image_latents: Optional[torch.Tensor] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            pooled_prompt_embeds: Optional[torch.Tensor] = None,
            prompt_attention_mask: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_attention_mask: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            prompt_template: Dict[str, Any] = DEFAULT_PROMPT_TEMPLATE,
            max_sequence_length: int = 256,
            one_frame_inference_options: Optional[List[str]] = None,  # ADDED
    ):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        self.check_inputs(prompt, prompt_2, height, width, prompt_embeds, callback_on_step_end_tensor_inputs,
                          prompt_template)
        has_neg_prompt = negative_prompt is not None or (
                    negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None)
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device
        transformer_dtype = self.transformer.dtype
        vae_dtype = self.vae.dtype
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # ---- ADDED: Single Frame Detection & Options ----
        is_one_frame_inference = num_frames == 1
        _one_frame_options_set = set(one_frame_inference_options) if one_frame_inference_options else set()
        # ---- END ADDED ----

        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.encode_prompt(prompt=prompt,
                                                                                        prompt_2=prompt_2,
                                                                                        prompt_template=prompt_template,
                                                                                        num_videos_per_prompt=num_videos_per_prompt,
                                                                                        prompt_embeds=prompt_embeds,
                                                                                        pooled_prompt_embeds=pooled_prompt_embeds,
                                                                                        prompt_attention_mask=prompt_attention_mask,
                                                                                        device=device,
                                                                                        dtype=transformer_dtype,
                                                                                        max_sequence_length=max_sequence_length)
        if do_true_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                prompt=negative_prompt, prompt_2=negative_prompt_2, prompt_template=prompt_template,
                num_videos_per_prompt=num_videos_per_prompt, prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                prompt_attention_mask=negative_prompt_attention_mask, device=device, dtype=transformer_dtype,
                max_sequence_length=max_sequence_length)

        _processed_image = self.video_processor.preprocess(image, height, width)
        image_embeds = self.encode_image(_processed_image, device=device, dtype=transformer_dtype)

        if last_image is not None and not is_one_frame_inference:  # MODIFIED: ignore last_image for single frame
            _processed_last_image = self.video_processor.preprocess(last_image, height, width)
            _last_image_embeds_only = self.encode_image(_processed_last_image, device=device, dtype=transformer_dtype)
            last_image_embeds = (image_embeds + _last_image_embeds_only) / 2
        else:
            last_image_embeds = None  # Ensure None if not used or in single frame mode

        num_channels_latents = self.transformer.config.in_channels

        # ---- MODIFIED: num_latent_sections calculation ----
        if is_one_frame_inference:
            num_latent_sections = 1
        else:
            window_num_frames_pixel = (latent_window_size - 1) * self.vae_scale_factor_temporal + 1
            num_latent_sections = max(1, (num_frames + window_num_frames_pixel - 1) // window_num_frames_pixel)
        # ---- END MODIFIED ----

        history_sizes = [1, 2, 16]
        history_latents = torch.zeros(batch_size * num_videos_per_prompt, num_channels_latents, sum(history_sizes),
                                      height // self.vae_scale_factor_spatial, width // self.vae_scale_factor_spatial,
                                      device=device, dtype=torch.float32)

        _accumulated_generated_latents = None  # Used to store output latents from DiT steps
        # total_generated_latent_frames_count = 0 # Not strictly needed if we handle _accumulated_generated_latents correctly

        image_latents = self.prepare_image_latents(_processed_image, dtype=torch.float32, device=device,
                                                   generator=generator, latents=image_latents)
        if last_image is not None and not is_one_frame_inference and last_image_latents is None:
            _prepared_last_image_latents = self.prepare_image_latents(_processed_last_image, dtype=torch.float32,
                                                                      device=device,
                                                                      generator=generator)  # Assign to a temp var
            if last_image_embeds is not None:  # Only assign if last_image was actually used for embeds
                last_image_latents = _prepared_last_image_latents
        elif is_one_frame_inference:  # Ensure last_image_latents is None for single frame
            last_image_latents = None

        # ---- MODIFIED: latent_paddings calculation ----
        if is_one_frame_inference:
            latent_paddings = [0]
        else:
            latent_paddings = list(reversed(range(num_latent_sections)))
            if num_latent_sections > 4:
                latent_paddings = [3] + [2] * (num_latent_sections - 3) + [1, 0]
        # ---- END MODIFIED ----

        guidance = torch.tensor([guidance_scale] * (batch_size * num_videos_per_prompt), dtype=transformer_dtype,
                                device=device) * 1000.0

        for k in range(num_latent_sections):
            is_first_section = k == 0
            is_last_section = k == num_latent_sections - 1  # Relative to the loop, not necessarily the video end
            latent_padding_size = latent_paddings[k] * latent_window_size

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, *history_sizes])).to(device)
            (indices_prefix, _indices_padding_unused, indices_latents_generation_window, indices_postfix,
             indices_latents_history_2x, indices_latents_history_4x) = indices.split(
                [1, latent_padding_size, latent_window_size, *history_sizes], dim=0)

            # ---- MODIFIED: Determine actual latent indices and frames for DiT this step ----
            if is_one_frame_inference:
                indices_latents = indices_latents_generation_window[:1]
                num_latent_frames_for_DIT_step = 1
            else:
                indices_latents = indices_latents_generation_window[:-1]
                num_latent_frames_for_DIT_step = latent_window_size - 1
            # ---- END MODIFIED ----

            _indices_clean_latents = torch.cat([indices_prefix, indices_postfix], dim=0)
            _latents_prefix_clean = image_latents
            _latents_postfix_clean, _latents_history_2x_clean, _latents_history_4x_clean = history_latents.split(
                history_sizes, dim=2)

            if last_image_latents is not None and is_first_section and not is_one_frame_inference:
                _latents_postfix_clean = last_image_latents

            # ---- MODIFIED: Adjust clean history for single-frame mode ----
            if is_one_frame_inference:
                if "no_post" in _one_frame_options_set:
                    _latents_postfix_clean = torch.zeros_like(_latents_postfix_clean)
                    _indices_clean_latents = indices_prefix
                else:
                    _latents_postfix_clean = torch.zeros_like(
                        _latents_postfix_clean)  # Default zero out for single frame
                if "no_2x" in _one_frame_options_set: _latents_history_2x_clean = torch.zeros_like(
                    _latents_history_2x_clean)
                if "no_4x" in _one_frame_options_set: _latents_history_4x_clean = torch.zeros_like(
                    _latents_history_4x_clean)
            # ---- END MODIFIED ----
            latents_clean = torch.cat([_latents_prefix_clean, _latents_postfix_clean], dim=2)

            # Prepare initial noise for this section's DIT generation
            latents = self.prepare_latents(batch_size * num_videos_per_prompt, num_channels_latents, height, width,
                                           num_latent_frames_for_DIT_step, dtype=torch.float32, device=device,
                                           generator=generator)

            _sigmas_for_scheduler = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1] if sigmas is None else sigmas
            _image_seq_len = (latents.shape[2] * latents.shape[3] * latents.shape[
                4]) / self.transformer.config.patch_size ** 2
            _exp_max = 7.0
            _mu = calculate_shift(_image_seq_len, self.scheduler.config.get("base_image_seq_len", 256),
                                  self.scheduler.config.get("max_image_seq_len", 4096),
                                  self.scheduler.config.get("base_shift", 0.5),
                                  self.scheduler.config.get("max_shift", 1.15))
            _mu = min(_mu, math.log(_exp_max))
            timesteps, current_num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device,
                                                                        sigmas=_sigmas_for_scheduler, mu=_mu)
            num_warmup_steps = len(timesteps) - current_num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)

            with self.progress_bar(total=current_num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt: continue
                    self._current_timestep = t
                    timestep = t.expand(latents.shape[0])
                    noise_pred = self.transformer(hidden_states=latents.to(transformer_dtype), timestep=timestep,
                                                  encoder_hidden_states=prompt_embeds,
                                                  encoder_attention_mask=prompt_attention_mask,
                                                  pooled_projections=pooled_prompt_embeds, image_embeds=image_embeds,
                                                  indices_latents=indices_latents, guidance=guidance,
                                                  latents_clean=latents_clean.to(transformer_dtype),
                                                  indices_latents_clean=_indices_clean_latents,
                                                  latents_history_2x=_latents_history_2x_clean.to(transformer_dtype),
                                                  indices_latents_history_2x=indices_latents_history_2x,
                                                  latents_history_4x=_latents_history_4x_clean.to(transformer_dtype),
                                                  indices_latents_history_4x=indices_latents_history_4x,
                                                  attention_kwargs=attention_kwargs, return_dict=False)[0]
                    if do_true_cfg:
                        neg_noise_pred = \
                        self.transformer(hidden_states=latents.to(transformer_dtype), timestep=timestep,
                                         encoder_hidden_states=negative_prompt_embeds,
                                         encoder_attention_mask=negative_prompt_attention_mask,
                                         pooled_projections=negative_pooled_prompt_embeds, image_embeds=image_embeds,
                                         indices_latents=indices_latents, guidance=guidance,
                                         latents_clean=latents_clean.to(transformer_dtype),
                                         indices_latents_clean=_indices_clean_latents,
                                         latents_history_2x=_latents_history_2x_clean.to(transformer_dtype),
                                         indices_latents_history_2x=indices_latents_history_2x,
                                         latents_history_4x=_latents_history_4x_clean.to(transformer_dtype),
                                         indices_latents_history_4x=indices_latents_history_4x,
                                         attention_kwargs=attention_kwargs, return_dict=False)[0]
                        noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    latents = self.scheduler.step(noise_pred.float(), t, latents, return_dict=False)[0]
                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for cb_k_tensor_input in callback_on_step_end_tensor_inputs: callback_kwargs[
                            cb_k_tensor_input] = locals()[cb_k_tensor_input]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)
                    if i == len(timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0): progress_bar.update()
                    if XLA_AVAILABLE: xm.mark_step()

            _generated_latents_this_section = latents.to(torch.float32)

            # ---- MODIFIED: Accumulate generated latents ----
            if is_one_frame_inference:
                _accumulated_generated_latents = _generated_latents_this_section  # This is the single generated frame
            else:  # Video mode
                if is_last_section and num_latent_sections > 1:  # Original: prepend start_latent for last section (which is video start)
                    _generated_latents_this_section = torch.cat([image_latents, _generated_latents_this_section], dim=2)

                if _accumulated_generated_latents is None:
                    _accumulated_generated_latents = _generated_latents_this_section
                else:
                    _accumulated_generated_latents = torch.cat(
                        [_generated_latents_this_section, _accumulated_generated_latents], dim=2)

            if not is_one_frame_inference:  # Update history_latents for next section (video mode only)
                if _generated_latents_this_section.shape[2] >= 1: history_latents[:, :,
                                                                  0:1] = _generated_latents_this_section[:, :, 0:1]
                if _generated_latents_this_section.shape[2] >= history_sizes[1]: history_latents[:, :,
                                                                                 history_sizes[0]:history_sizes[0] +
                                                                                                  history_sizes[
                                                                                                      1]] = _generated_latents_this_section[
                                                                                                            :, :, :
                                                                                                                  history_sizes[
                                                                                                                      1]]
                if _generated_latents_this_section.shape[2] >= history_sizes[2]: history_latents[:, :,
                                                                                 history_sizes[0] + history_sizes[
                                                                                     1]:] = _generated_latents_this_section[
                                                                                            :, :, :history_sizes[2]]
            # ---- END MODIFIED ----

        self._current_timestep = None
        # `_accumulated_generated_latents` now holds the final latents (single frame or full video sequence)

        if not output_type == "latent":
            _final_latents_for_decode = _accumulated_generated_latents.to(vae_dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(_final_latents_for_decode, return_dict=False)[0]
            # `num_frames` is the original target pixel frames for video, or 1 for image
            video = self.video_processor.postprocess_video(video, output_type=output_type, num_frames=num_frames)
        else:
            video = [_accumulated_generated_latents]

        self.maybe_free_model_hooks()
        if not return_dict: return (video,)
        return HunyuanVideoFramepackPipelineOutput(frames=video)

    def _soft_append(self, history: torch.Tensor, current: torch.Tensor, overlap: int = 0):
        if overlap <= 0: return torch.cat([history, current], dim=2)
        assert history.shape[2] >= overlap, f"Current length ({history.shape[2]}) must be >= overlap ({overlap})"
        assert current.shape[2] >= overlap, f"History length ({current.shape[2]}) must be >= overlap ({overlap})"
        weights = torch.linspace(1, 0, overlap, dtype=history.dtype, device=history.device).view(1, 1, -1, 1, 1)
        blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
        output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)
        return output.to(history)