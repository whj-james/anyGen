import unittest

import PIL
import requests
import asyncio
from io import BytesIO

from services import img2img
from schemas import Image2Image
from common.config import CACHE_DIR


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


class TestGenImg(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        return super().setUpClass()
    
    async def test_img2img(self):
        img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
        init_image = download_image(img_url).resize((512, 512))
        img2img_params = Image2Image(
            img_prompt=init_image,
            str_prompt='Face of a yellow cat, high resolution, sitting on a park bench'
        )
        gen_image = await img2img(img2img_params)
        gen_image.save(CACHE_DIR.joinpath(f'cat.png'))
