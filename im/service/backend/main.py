from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import StreamingResponse
import fastapi as _fapi

from typing_extensions import Annotated
from PIL import Image

import schemas as _schemas
import services as _services
import txt2img as _txt2img
import io


app = FastAPI()


# Endpoint to test the Front-end and backend
@app.get("/hi")
async def root():
    return {"message": "Hi"}

@app.post("/api/generate/img2img/")
async def img2img(
    img_prompt: Annotated[UploadFile, File()],
    str_prompt: Annotated[str, Form()],
    sampling_steps: Annotated[int, Form(ge=1.0, le=150, alias='inference_steps')] = 10,
    cfg_scale: Annotated[float, Form(ge=1.0, le=30.0, alias='guidance_scale')] = 7.5,
    denoising_strength: Annotated[float, Form(ge=0.0, le=1.0, alias='strength')] = 0.5,
    num_images_per_prompt: Annotated[int, Form()] = 4,
    ):
    img2img_params = _schemas.Image2Image(
        img_prompt=Image.open(io.BytesIO(await img_prompt.read())).convert('RGB'),
        str_prompt=str_prompt,
        num_inference_steps=sampling_steps,
        guidance_scale=cfg_scale,
        strength=denoising_strength,
        num_images_per_prompt=num_images_per_prompt,
        )
    image = await _services.img2img(create_params=img2img_params)

    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")


@app.post("/api/generate/txt2img/")
async def txt2img(
    str_prompt: Annotated[str, Form()],
    sampling_steps: Annotated[int, Form(ge=1.0, le=150, alias='inference_steps')] = 10,
    cfg_scale: Annotated[float, Form(ge=1.0, le=30.0, alias='guidance_scale')] = 7.5,
    num_images_per_prompt: Annotated[int, Form()] = 4,
    ):
    txt2img_params = _schemas.Text2Image(
        str_prompt=str_prompt,
        num_inference_steps=sampling_steps,
        guidance_scale=cfg_scale,
        num_images_per_prompt=num_images_per_prompt,
        )
    image = await _txt2img.txt2img(create_params=txt2img_params)

    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")