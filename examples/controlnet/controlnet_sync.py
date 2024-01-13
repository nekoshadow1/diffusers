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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import os
from typing import Any, Callable, List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config

from diffusers.loaders import FromOriginalControlnetMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.embeddings import TextImageProjection, TextImageTimeEmbedding, TextTimeEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unet_2d_blocks import CrossAttnDownBlock2D, DownBlock2D, UNetMidBlock2D, UNetMidBlock2DCrossAttn, get_down_block
from diffusers.models.unet_2d_condition import UNet2DConditionModel

from diffusers.utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    MIN_PEFT_VERSION,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    _add_variant,
    _get_model_file,
    check_peft_version,
    deprecate,
    is_accelerate_available,
    is_torch_version,
    logging,
)
from diffusers.utils.hub_utils import PushToHubMixin

from SyncDreamer.ldm.modules.attention import default, zero_module, checkpoint
from SyncDreamer.ldm.modules.diffusionmodules.openaimodel import UNetModel
from SyncDreamer.ldm.modules.diffusionmodules.util import timestep_embedding
from SyncDreamer.ldm.models.diffusion.sync_dreamer_attention import DepthWiseAttention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class DepthAttention(nn.Module):
    def __init__(self, query_dim, context_dim, heads, dim_head, output_bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Conv2d(query_dim, inner_dim, 1, 1, bias=False)
        self.to_k = nn.Conv3d(context_dim, inner_dim, 1, 1, bias=False)
        self.to_v = nn.Conv3d(context_dim, inner_dim, 1, 1, bias=False)
        if output_bias:
            self.to_out = nn.Conv2d(inner_dim, query_dim, 1, 1)
        else:
            self.to_out = nn.Conv2d(inner_dim, query_dim, 1, 1, bias=False)

    def forward(self, x, context):
        """

        @param x:        b,f0,h,w
        @param context:  b,f1,d,h,w
        @return:
        """
        hn, hd = self.heads, self.dim_head
        b, _, h, w = x.shape
        b, _, d, h, w = context.shape

        q = self.to_q(x).reshape(b,hn,hd,h,w) # b,t,h,w
        k = self.to_k(context).reshape(b,hn,hd,d,h,w) # b,t,d,h,w
        v = self.to_v(context).reshape(b,hn,hd,d,h,w) # b,t,d,h,w

        sim = torch.sum(q.unsqueeze(3) * k, 2) * self.scale # b,hn,d,h,w
        attn = sim.softmax(dim=2)

        # b,hn,hd,d,h,w * b,hn,1,d,h,w
        out = torch.sum(v * attn.unsqueeze(2), 3) # b,hn,hd,h,w
        out = out.reshape(b,hn*hd,h,w)
        return self.to_out(out)


class DepthTransformer(nn.Module):
    def __init__(self, dim, n_heads, d_head, context_dim=None, checkpoint=False):
        super().__init__()
        inner_dim = n_heads * d_head
        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, inner_dim, 1, 1),
            nn.GroupNorm(8, inner_dim),
            nn.SiLU(True),
        )
        self.proj_context = nn.Sequential(
            nn.Conv3d(context_dim, context_dim, 1, 1, bias=False), # no bias
            nn.GroupNorm(8, context_dim),
            nn.ReLU(True), # only relu, because we want input is 0, output is 0
        )
        self.depth_attn = DepthAttention(query_dim=inner_dim, heads=n_heads, dim_head=d_head, context_dim=context_dim, output_bias=False)  # is a self-attention if not self.disable_self_attn
        self.proj_out = nn.Sequential(
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, 1, bias=False),
            nn.GroupNorm(8, inner_dim),
            nn.ReLU(True),
            zero_module(nn.Conv2d(inner_dim, dim, 3, 1, 1, bias=False)),
        )
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context):
        x_in = x
        x = self.proj_in(x)
        context = self.proj_context(context)
        x = self.depth_attn(x, context)
        x = self.proj_out(x) + x_in
        return x

@dataclass
class ControlNetOutputSync(BaseOutput):
    """
    The output of [`ControlNetModelSync`].

    Args:
        down_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
        mid_down_block_re_sample (`torch.Tensor`):
            The activation of the midde block (the lowest sample resolution). Each tensor should be of shape
            `(batch_size, channel * lowest_resolution, height // lowest_resolution, width // lowest_resolution)`.
            Output can be used to condition the original UNet's middle block activation.
    """

    down_block_res_samples: Tuple[torch.Tensor]
    mid_block_res_sample: torch.Tensor


class ControlNetConditioningEmbeddingSync(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class ControlNetModelSync(UNetModel, ModelMixin, ConfigMixin):
    use_fp16 = False
    dtype = torch.float16 if use_fp16 else torch.float32
        
    @register_to_config
    def __init__(
        self,
        volume_dims=[64, 128, 256, 512], 
        image_size=32,
        in_channels=8,
        model_channels=320,
        out_channels=4,
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        channel_mult=[1, 2, 4, 4],
        use_checkpoint=False,
        legacy=False,
        num_heads=8,
        use_spatial_transformer=True,
        transformer_depth=1,
        context_dim=768,
    ):
        
        super().__init__(image_size=image_size, in_channels=in_channels, model_channels=model_channels, out_channels=out_channels, num_res_blocks=num_res_blocks, attention_resolutions=attention_resolutions, channel_mult=channel_mult, use_checkpoint=use_checkpoint, legacy=legacy, num_heads=num_heads, use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth, context_dim=context_dim)
        
        block_out_channels = (320, 640, 1280, 1280)
        conditioning_embedding_out_channels = (16, 32, 96, 256)
        conditioning_channels = 3
        down_block_types = (
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        )
        layers_per_block = 2
        
        # input
        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
            
        d0,d1,d2,d3 = volume_dims

        # 4
        ch = model_channels*channel_mult[2]
        self.middle_conditions = DepthTransformer(ch, 4, d3 // 2, context_dim=d3)

        self.controlnet_cond_embedding = ControlNetConditioningEmbeddingSync(
            conditioning_embedding_channels=self.in_channels,
            block_out_channels=conditioning_embedding_out_channels,
            conditioning_channels=conditioning_channels,
        )
        
        self.controlnet_down_blocks = nn.ModuleList([])
        # down
        output_channel = block_out_channels[0]

        controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_down_blocks.append(controlnet_block)
        
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            
            for _ in range(layers_per_block):
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

            if not is_final_block:
                controlnet_block = nn.Conv2d(output_channel, output_channel, kernel_size=1)
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
        
        # mid
        mid_block_channel = block_out_channels[-1]
        
        controlnet_block = nn.Conv2d(mid_block_channel, mid_block_channel, kernel_size=1)
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block
        
    @classmethod
    def from_unet(
        cls,
        unet: DepthWiseAttention,
        load_weights_from_unet: bool = True,
    ):
        r"""
        Instantiate a [`ControlNetModelSync`] from [`DepthWiseAttention`].

        Parameters:
            unet (`DepthWiseAttention`):
                The UNet model weights to copy to the [`ControlNetModelSync`]. All configuration options are also copied
                where applicable.
        """

        controlnet = cls(
            image_size=32, 
            in_channels=8, 
            model_channels=320, 
            out_channels=4, 
            num_res_blocks=2,
            attention_resolutions=[ 4, 2, 1 ],
            num_heads=8,
            volume_dims=[64, 128, 256, 512],
            channel_mult=[ 1, 2, 4, 4 ],
            use_spatial_transformer=True,
            transformer_depth=1,
            context_dim=768,
            use_checkpoint=False,
            legacy=False,
        )

        if load_weights_from_unet:
            controlnet.time_embed.load_state_dict(unet.time_embed.state_dict())
            controlnet.input_blocks.load_state_dict(unet.input_blocks.state_dict())
            controlnet.middle_block.load_state_dict(unet.middle_block.state_dict())
            controlnet.middle_conditions.load_state_dict(unet.middle_conditions.state_dict())

        return controlnet

    def forward(self, x, timesteps=None, controlnet_cond=None, conditioning_scale=1.0, context=None, return_dict = True, source_dict=None, **kwargs):

        # 1-4. Down and mid blocks, incluidng time embedding
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)   
        emb = self.time_embed(t_emb)
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        x = x + controlnet_cond 
        h = x.type(self.dtype)
        for index, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            hs.append(h)
        
        h = self.middle_block(h, emb, context)
        h = self.middle_conditions(h, context=source_dict[h.shape[-1]])

        # 5. Control net blocks
        controlnet_down_block_res_samples = ()
        
        assert len(hs) == len(self.controlnet_down_blocks), "Number of layers in 'hs' should be equal to 'controlnet_down_blocks'"
        
        for down_block_res_sample, controlnet_block in zip(hs, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(h)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)

        return ControlNetOutputSync(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
