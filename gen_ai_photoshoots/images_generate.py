from diffusers import StableDiffusionPipeline
import torch


def generate_image(prompt):
    pipe = StableDiffusionPipeline.from_pretrained(
        "./pipeline-folder",
        torch_dtype=torch.float16,
    ).to("cuda")
    guidance_scale = 9
    image = pipe(prompt, guidance_scale=guidance_scale).images
    return image
