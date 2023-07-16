import pydantic as _pydantic
from typing import Optional
from PIL.Image import Image


class _SdParamsBase(_pydantic.BaseModel):
    num_inference_steps: int = 10  # sampling steps
    guidance_scale: float = 7.5  # also called CFG scale
    num_images_per_prompt: Optional[int] = 4
    strength: float = 0.5


class Image2Image(_SdParamsBase):
    str_prompt: str
    img_prompt: object
