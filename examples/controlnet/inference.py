import argparse
import os
import torch
import numpy as np
import cv2
from transformers import pipeline
from diffusers.utils import load_image, make_image_grid
from diffusers import UniPCMultistepScheduler
from pipeline_controlnet_sync import StableDiffusionControlNetPipeline
from controlnet_sync import ControlNetModelSync
from omegaconf import OmegaConf

from SyncDreamer.ldm.util import instantiate_from_config, prepare_inputs

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Script to create training data for controlnet+syncdreamer.")
    parser.add_argument(
        "--MODEL_PATH",
        type=str,
        default="trained_model/",
        required=False,
        help="Path to trained controlnet model.",
    )
    parser.add_argument(
        "--INPUT_PATH",
        type=str,
        default=None,
        required=True,
        help="Path to input image.",
    )
    parser.add_argument(
        "--AZIMUTH",
        type=float,
        default=None,
        required=True,
        help="Input azimuth.",
    )
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

args = parse_args()

def check_inputs():
    if args.AZIMUTH < 0:
        raise Exception('Please input a positive azimuth!')
    if not os.path.isfile(args.INPUT_PATH):
        raise Exception('Please use a valid input image!')
    
check_inputs()

def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cuda')
    model.load_state_dict(ckpt['state_dict'],strict=strict)
    model = model.cuda().eval()
    return model

controlnet = ControlNetModelSync.from_pretrained(args.MODEL_PATH, torch_dtype=torch.float32, use_safetensors=True)
cfg = 'SyncDreamer/configs/syncdreamer.yaml'
dreamer = load_model(cfg, 'SyncDreamer/ckpt/syncdreamer-pretrain.ckpt', strict=True)

controlnet.to('cuda', dtype=torch.float32)
dreamer.to('cuda', dtype=torch.float32)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, dreamer=dreamer, torch_dtype=torch.float32, use_safetensors=True
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images) : return images, [False for i in images]

# edit the StableDiffusionSafetyChecker class so that, when called, it just returns the images and an array of True values
safety_checker.StableDiffusionSafetyChecker.forward = sc

pipe.to('cuda', dtype=torch.float32)

output = pipe(
    '', image=None, conditioning_image_path=args.INPUT_PATH,
)

targe_index = round(args.AZIMUTH / 22.5)

OUTPUT_PATH = 'output/'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

cv2.imwrite(OUTPUT_PATH + os.path.basename(args.INPUT_PATH).split('/')[-1], output[target_index])
