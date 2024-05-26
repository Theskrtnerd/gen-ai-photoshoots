from diffusers import StableDiffusionPipeline
import torch


def generate_image(prompt):
    """
    Generate an image based on the provided prompt using the Stable Diffusion model.

    Args:
        prompt (str): The text prompt describing the desired image.

    Returns:
        PIL.Image: The generated image based on the prompt.

    This function performs the following steps:
        1. Loads the Stable Diffusion pipeline from the pretrained model directory.
        2. Moves the pipeline to the GPU for faster processing.
        3. Sets the guidance scale to influence the strength of the prompt in the generation process.
        4. Generates the image using the provided prompt.
        5. Returns the generated image.
    """
    pipe = StableDiffusionPipeline.from_pretrained(
        "./pipeline-folder",
        torch_dtype=torch.float16,
    ).to("cuda")
    guidance_scale = 9
    image = pipe(prompt, guidance_scale=guidance_scale).images
    return image
