"""
Adapted from https://huggingface.co/spaces/stabilityai/stable-diffusion
"""

import torch

import time

import gradio as gr

from constants import css, examples, img_height, img_width, num_images_to_gen
from share_btn import community_icon_html, loading_icon_html, share_js

from diffusers import StableDiffusionPanoramaPipeline, DDIMScheduler

model_ckpt = "stabilityai/stable-diffusion-2-base"
scheduler = DDIMScheduler.from_pretrained(model_ckpt, subfolder="scheduler")
pipe = StableDiffusionPanoramaPipeline.from_pretrained(
     model_ckpt, scheduler=scheduler, torch_dtype=torch.float16
)

pipe = pipe.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def generate_image_fn(prompt: str, guidance_scale: float) -> list:
    start_time = time.time()
    prompt = "a photo of the dolomites"
    image = pipe(prompt, guidance_scale=guidance_scale).images
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds.")
    return image


description = "This Space demonstrates MultiDiffusion Text2Panorama using Stable Diffusion model. You can use it for generating custom pokemons. To get started, either enter a prompt and pick one from the examples below. For details on the fine-tuning procedure, refer to [this repository]()."
article = "This Space leverages a T4 GPU to run the predictions. We use mixed-precision to speed up the inference latency."
gr.Interface(
    generate_image_fn,
    inputs=[
        gr.Textbox(
            label="Enter your prompt",
            max_lines=1,
            placeholder="a photo of the dolomites",
        ),
        gr.Slider(value=40, minimum=8, maximum=50, step=1),
    ],
    outputs=gr.Gallery().style(grid=[2], height="auto"),
    title="Generate custom pokemons",
    description=description,
    article=article,
    examples=[["a photo of the dolomites", 40]],
    allow_flagging=False,
).launch(enable_queue=True)