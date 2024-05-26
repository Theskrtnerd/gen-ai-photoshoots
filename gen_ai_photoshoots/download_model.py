import os


def download_model(my_bar):
    """
    Download and save the pretrained Stable Diffusion model components.

    This function downloads the tokenizer, text encoder, variational autoencoder (VAE),
    U-Net, and feature extractor components of the Stable Diffusion model and saves them
    to a specified directory. It also updates a progress bar to reflect the download progress.

    Args:
        my_bar: A Streamlit progress bar object used to update the download progress.

    The function performs the following steps:
        1. Checks if the save directory exists; if not, creates it.
        2. Downloads the tokenizer and updates the progress bar to 20%.
        3. Downloads the text encoder and updates the progress bar to 40%.
        4. Downloads the VAE and updates the progress bar to 60%.
        5. Downloads the U-Net and updates the progress bar to 80%.
        6. Downloads the feature extractor and updates the progress bar to 96%.
        7. Saves each downloaded component to the specified directory.
    """
    save_directory = "./stable_diffusion_models"

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
        my_bar.progress(0, "Downloading the pretrained model...")
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPFeatureExtractor
        from diffusers import AutoencoderKL, UNet2DConditionModel

        # The Stable Diffusion checkpoint we'll fine-tune
        model_id = "CompVis/stable-diffusion-v1-4"
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        my_bar.progress(20, "Downloading the pretrained model...")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
        my_bar.progress(40, "Downloading the pretrained model...")
        vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        my_bar.progress(60, "Downloading the pretrained model...")
        unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        my_bar.progress(80, "Downloading the pretrained model...")
        feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        my_bar.progress(96, "Downloading the pretrained model...")

        # Save each component
        tokenizer.save_pretrained(f"{save_directory}/tokenizer")
        text_encoder.save_pretrained(f"{save_directory}/text_encoder")
        vae.save_pretrained(f"{save_directory}/vae")
        unet.save_pretrained(f"{save_directory}/unet")
        feature_extractor.save_pretrained(f"{save_directory}/feature_extractor")
