import os


def model_download():
    save_directory = "./stable_diffusion_models"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor
        from diffusers import AutoencoderKL, UNet2DConditionModel

        # The Stable Diffusion checkpoint we'll fine-tune
        model_id = "CompVis/stable-diffusion-v1-4"
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")

        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

        # Save each component
        tokenizer.save_pretrained(f"{save_directory}/tokenizer")
        text_encoder.save_pretrained(f"{save_directory}/text_encoder")
        vae.save_pretrained(f"{save_directory}/vae")
        unet.save_pretrained(f"{save_directory}/unet")
        feature_extractor.save_pretrained(f"{save_directory}/feature_extractor")
