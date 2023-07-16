from pathlib import Path 
import schemas as _schemas

import torch 
from diffusers import StableDiffusionImg2ImgPipeline
from PIL.Image import Image
import os
from dotenv import load_dotenv

from common.config import CACHE_DIR

load_dotenv()

# Get the token from HuggingFace 
"""
Note: make sure .env exist and contains your token
"""
HF_TOKEN = os.getenv('HF_TOKEN')

# Create the pipe 
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN,
    cache_dir=CACHE_DIR
    )

if torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cuda" if torch.cuda.is_available() else "cpu"

pipe.to(device)


async def img2img(create_params: _schemas.Image2Image) -> Image: 
    image: Image = pipe(create_params.str_prompt,
                        image=create_params.img_prompt,
                        guidance_scale=create_params.guidance_scale, 
                        num_inference_steps=create_params.num_inference_steps, 
                        num_images_per_prompt=create_params.num_images_per_prompt,
                        strength=create_params.strength,
                    ).images[0]
    
    return image
