from typing import Iterable
import schemas as _schemas

import torch 
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import os
from dotenv import load_dotenv

from common.config import CACHE_DIR
from diffusers import StableDiffusionControlNetImg2ImgPipeline, DiffusionPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionPipeline
from diffusers.utils import load_image
import numpy as np
import torch
import safetensors

import cv2

load_dotenv()

# Get the token from HuggingFace 
"""
Note: make sure .env exist and contains your token
"""
HF_TOKEN = os.getenv('HF_TOKEN')

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    revision="fp16", 
    controlnet=controlnet, 
    use_auth_token=HF_TOKEN,
    torch_dtype=torch.float16,
    cache_dir=CACHE_DIR,
    safety_checker=None,
    requires_safety_checker=False
)
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

if torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cuda" if torch.cuda.is_available() else "cpu"

pipe.to(device)


async def img2img(create_params: _schemas.Image2Image) -> Image.Image:
    img_prompt: Image.Image = resize_to(create_params.img_prompt, 400, 600)
    canny_image = canny(np.array(img_prompt), 70, 200)
    image: Image.Image = pipe(create_params.str_prompt,
                        image=img_prompt,
                        guidance_scale=create_params.guidance_scale, 
                        strength=create_params.strength,
                        control_image=canny_image,
                        controlnet_conditioning_scale=0.45,
                        generator=torch.manual_seed(create_params.seed) if create_params.seed is not None else None
                    ).images[0]
    
    return image

def canny(image, lower, upper):
    # get canny image
    canny_np_image = cv2.Canny(image, lower, upper)
    canny_np_image = canny_np_image[:, :, None]
    canny_np_image = np.concatenate([canny_np_image, canny_np_image, canny_np_image], axis=2)
    canny_image = Image.fromarray(canny_np_image)
    return canny_image

def resize_with_padding(image, target_width, target_height, padding_color=(255, 255, 255)): 
     
    height, width = image.shape[:2]
    aspect_ratio = width / float(height)
    target_aspect_ratio = target_width / float(target_height)

    if target_aspect_ratio > aspect_ratio:
        new_width = target_height * aspect_ratio
        new_height = target_height
    else:
        new_width = target_width
        new_height = target_width / aspect_ratio

    pad_width = int((target_width - new_width) / 2)
    pad_height = int((target_height - new_height) / 2)

    resized_image = cv2.resize(image, (int(new_width), int(new_height)), interpolation=cv2.INTER_CUBIC)
    canvas = np.ones((target_height, target_width, 3), dtype=np.uint8) * np.array(padding_color, dtype=np.uint8)
    canvas[pad_height:pad_height+resized_image.shape[0], pad_width:pad_width+resized_image.shape[1]] = resized_image

    return canvas


def resize_to(image, target_width, target_height):
    np_image = np.array(image)
    # print(np_image.shape)

    if np_image.shape[0] > np_image.shape[1]:
        resized_np_image = resize_with_padding(np_image, target_width, target_height)
    else:
        resized_np_image = resize_with_padding(np_image, target_height, target_width)

    resized_image = Image.fromarray(resized_np_image)
    return resized_image
