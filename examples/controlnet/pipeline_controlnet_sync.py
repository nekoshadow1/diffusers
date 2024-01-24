# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, ImageProjection, UNet2DConditionModel
from controlnet_sync import ControlNetModelSync

from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import is_compiled_module, is_torch_version, randn_tensor
# from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from pipeline_utils_sync import DiffusionPipeline

from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from SyncDreamer.ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion, SyncDDIMSampler
from SyncDreamer.ldm.util import prepare_inputs

from tqdm import tqdm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> # !pip install opencv-python transformers accelerate
        >>> from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
        >>> from diffusers.utils import load_image
        >>> import numpy as np
        >>> import torch

        >>> import cv2
        >>> from PIL import Image

        >>> # download an image
        >>> image = load_image(
        ...     "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
        ... )
        >>> image = np.array(image)

        >>> # get canny image
        >>> image = cv2.Canny(image, 100, 200)
        >>> image = image[:, :, None]
        >>> image = np.concatenate([image, image, image], axis=2)
        >>> canny_image = Image.fromarray(image)

        >>> # load control net and stable diffusion v1-5
        >>> controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        >>> pipe = StableDiffusionControlNetPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        ... )

        >>> # speed up diffusion process with faster scheduler and memory optimization
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        >>> # remove following line if xformers is not installed
        >>> pipe.enable_xformers_memory_efficient_attention()

        >>> pipe.enable_model_cpu_offload()

        >>> # generate image
        >>> generator = torch.manual_seed(0)
        >>> image = pipe(
        ...     "futuristic-looking woman", num_inference_steps=20, generator=generator, image=canny_image
        ... ).images[0]
        ```
"""

class StableDiffusionControlNetPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, IPAdapterMixin, FromSingleFileMixin
):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights
        - [`~loaders.FromSingleFileMixin.from_single_file`] for loading `.ckpt` files
        - [`~loaders.IPAdapterMixin.load_ip_adapter`] for loading IP Adapters

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the `unet` during the denoising process. If you set multiple
            ControlNets as a list, the outputs from each ControlNet are added together to create one combined
            additional conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for more details
            about a model's potential harms.
        feature_extractor ([`~transformers.CLIPImageProcessor`]):
            A `CLIPImageProcessor` to extract features from generated images; used as inputs to the `safety_checker`.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->unet->vae"
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        dreamer: SyncMultiviewDiffusion,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        self.register_modules(
            controlnet=controlnet,
            dreamer = dreamer,
        )
        
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        conditioning_image_path = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeine class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )
 
        def drop(cond, mask):
            shape = cond.shape
            B = shape[0]
            cond = mask.view(B,*[1 for _ in range(len(shape)-1)]) * cond
            return cond

        def get_drop_scheme(B, device):
            drop_scheme = 'default'
            if drop_scheme=='default':
                random = torch.rand(B, dtype=torch.float32, device=device)
                drop_clip = (random > 0.15) & (random <= 0.2)
                drop_volume = (random > 0.1) & (random <= 0.15)
                drop_concat = (random > 0.05) & (random <= 0.1)
                drop_all = random <= 0.05
            else:
                raise NotImplementedError
            return drop_clip, drop_volume, drop_concat, drop_all

        def unet_wrapper_forward(x, t, clip_embed, volume_feats, x_concat, is_train=False):
            drop_conditions = False
            if drop_conditions and is_train:
                B = x.shape[0]
                drop_clip, drop_volume, drop_concat, drop_all = get_drop_scheme(B, x.device)

                clip_mask = 1.0 - (drop_clip | drop_all).float()
                clip_embed = drop(clip_embed, clip_mask)

                volume_mask = 1.0 - (drop_volume | drop_all).float()
                for k, v in volume_feats.items():
                    volume_feats[k] = drop(v, mask=volume_mask)

                concat_mask = 1.0 - (drop_concat | drop_all).float()
                x_concat = drop(x_concat, concat_mask)

            use_zero_123 = True
            if use_zero_123:
                # zero123 does not multiply this when encoding, maybe a bug for zero123
                first_stage_scale_factor = 0.18215
                x_concat_ = x_concat * 1.0
                x_concat_[:, :4] = x_concat_[:, :4] / first_stage_scale_factor
            else:
                x_concat_ = x_concat

            x = torch.cat([x, x_concat_], 1)

            return x, t, clip_embed, volume_feats

        def unet_wrapper_forward_unconditional(x, t, clip_embed, volume_feats, x_concat):
            """

            @param x:             B,4,H,W
            @param t:             B,
            @param clip_embed:    B,M,768
            @param volume_feats:  B,C,D,H,W
            @param x_concat:      B,C,H,W
            @param is_train:
            @return:
            """
            x_ = torch.cat([x] * 2, 0)
            t_ = torch.cat([t] * 2, 0)
            clip_embed_ = torch.cat([clip_embed, torch.zeros_like(clip_embed)], 0)

            v_ = {}
            for k, v in volume_feats.items():
                v_[k] = torch.cat([v, torch.zeros_like(v)], 0)

            x_concat_ = torch.cat([x_concat, torch.zeros_like(x_concat)], 0)
            use_zero_123 = True
            if use_zero_123:
                # zero123 does not multiply this when encoding, maybe a bug for zero123
                first_stage_scale_factor = 0.18215
                x_concat_[:, :4] = x_concat_[:, :4] / first_stage_scale_factor
            x_ = torch.cat([x_, x_concat_], 1)
            return x_, t_, clip_embed_, v_
        
        def repeat_to_batch(tensor, B, VN):
            t_shape = tensor.shape
            ones = [1 for _ in range(len(t_shape)-1)]
            tensor_new = tensor.view(B,1,*t_shape[1:]).repeat(1,VN,*ones).view(B*VN,*t_shape[1:])
            return tensor_new

        flags_input = conditioning_image_path
        flags_sample_steps = 50
        weight_dtype = torch.float32
        
        data = prepare_inputs(flags_input, 30, -1)
                
        for k, v in data.items():
            data[k] = v.unsqueeze(0).cuda()
            data[k] = torch.repeat_interleave(data[k], repeats=1, dim=0)
            
        sampler = SyncDDIMSampler(self.dreamer, flags_sample_steps)

        data["conditioning_pixel_values"] = data['input_image']
        _, clip_embed, input_info = self.dreamer.prepare(data)
        controlnet_image = data["conditioning_pixel_values"].to(dtype=weight_dtype)
        controlnet_image = controlnet_image.permute(0, 3, 1, 2) # B, c, h, w
            
        image_size = 256
        latent_size = image_size//8
        C, H, W = 4, latent_size, latent_size
        B = clip_embed.shape[0]
        N = 16
        device = 'cuda'
        x_target_noisy = torch.randn([B, N, C, H, W], device=device)

        timesteps = sampler.ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        
        for i, step in enumerate(iterator):
            index = total_steps - i - 1 # index in ddim state
            is_step0=index==0
            
            time_steps = torch.full((B,), step, device=device, dtype=torch.long)

            x_input, elevation_input = input_info['x'], input_info['elevation']

            B, N, C, H, W = x_target_noisy.shape

            # construct source data
            v_embed = self.dreamer.get_viewpoint_embedding(B, elevation_input) # B,N,v_dim
            t_embed = self.dreamer.embed_time(time_steps)  # B,t_dim
            spatial_volume = self.dreamer.spatial_volume.construct_spatial_volume(x_target_noisy, t_embed, v_embed, self.dreamer.poses, self.dreamer.Ks)

            cfg_scale = 2.0
            unconditional_scale = cfg_scale
            batch_view_num = 4

            e_t = []
            target_indices = torch.arange(N) # N
            for ni in range(0, N, batch_view_num):
                x_target_noisy_ = x_target_noisy[:, ni:ni + batch_view_num]
                VN = x_target_noisy_.shape[1]
                x_target_noisy_ = x_target_noisy_.reshape(B*VN,C,H,W)

                time_steps_ = repeat_to_batch(time_steps, B, VN)
                target_indices_ = target_indices[ni:ni+batch_view_num].unsqueeze(0).repeat(B,1)
                clip_embed_, volume_feats_, x_concat_ = self.dreamer.get_target_view_feats(x_input, spatial_volume, clip_embed, t_embed, v_embed, target_indices_)

                if unconditional_scale!=1.0:
                    x_, t_, clip_embed_, volume_feats_ = unet_wrapper_forward_unconditional(x_target_noisy_, time_steps_, clip_embed_, volume_feats_, x_concat_)
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        x=x_,
                        timesteps=t_,
                        controlnet_cond=controlnet_image,
                        conditioning_scale=1.0,
                        context=clip_embed_,
                        return_dict=False,
                        source_dict=volume_feats_,
                    )

                    noise, s_uc = self.dreamer.model.diffusion_model(x_, t_, clip_embed_, down_block_res_samples, mid_block_res_sample, source_dict=volume_feats_).chunk(2)
                    noise = s_uc + unconditional_scale * (noise - s_uc)

                else:
                    x_noisy_, timesteps, clip_embed, volume_feats = unet_wrapper_forward(x_target_noisy_, time_steps_, clip_embed_, volume_feats_, x_concat_, is_train=False)
                    down_block_res_samples, mid_block_res_sample = controlnet(
                        x=x_noisy_,
                        timesteps=timesteps,
                        controlnet_cond=controlnet_image,
                        conditioning_scale=1.0,
                        context=clip_embed,
                        return_dict=False,
                        source_dict=volume_feats,
                    )

                    noise = self.dreamer.model.diffusion_model(x_noisy_, timesteps, clip_embed, down_block_res_samples, mid_block_res_sample, source_dict=volume_feats)

                e_t.append(noise.view(B,VN,4,H,W))

            e_t = torch.cat(e_t, 1)
            x_target_noisy = sampler.denoise_apply_impl(x_target_noisy, index, e_t, is_step0)
            
        N = x_target_noisy.shape[1]
        x_sample = torch.stack([self.dreamer.decode_first_stage(x_target_noisy[:, ni]) for ni in range(N)], 1)

        B, N, _, H, W = x_sample.shape
        x_sample = (torch.clamp(x_sample,max=1.0,min=-1.0) + 1) * 0.5
        x_sample = x_sample.permute(0,1,3,4,2).cpu().numpy() * 255
        x_sample = x_sample.astype(np.uint8)

        return x_sample[0, :, :, :, :]
