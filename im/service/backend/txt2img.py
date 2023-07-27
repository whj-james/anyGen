from typing import Iterable
import schemas as _schemas

import torch 
from diffusers import StableDiffusionPipeline
from PIL.Image import Image
import os
from dotenv import load_dotenv

load_dotenv()

# Get the token from HuggingFace 
"""
Note: make sure .env exist and contains your token
"""
HF_TOKEN = os.getenv('HF_TOKEN')

# Create the pipe 
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN,
    safety_checker = None,
    requires_safety_checker = False
    )

if torch.backends.mps.is_available():
    device = "mps"
else: 
    device = "cuda" if torch.cuda.is_available() else "cpu"

pipe.to(device)


async def txt2img(create_params: _schemas.Text2Image) -> Iterable[Image]: 
    image: Image = pipe(create_params.str_prompt,
                        guidance_scale=create_params.guidance_scale, 
                        num_inference_steps=create_params.num_inference_steps, 
                    ).images[0]
    
    return image