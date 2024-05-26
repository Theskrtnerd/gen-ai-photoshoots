from argparse import Namespace
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import math
from tqdm.auto import tqdm
from datasets import load_dataset
from gen_ai_photoshoots.download_model import download_model


class DreamBoothDataset(Dataset):
    """
    A custom dataset class for preparing data for training the Stable Diffusion model.

    Args:
        dataset (Dataset): The dataset containing images.
        instance_prompt (str): The prompt describing the instance in the images.
        tokenizer (CLIPTokenizer): The tokenizer for encoding text prompts.
        size (int, optional): The size to which images are resized. Default is 512.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(index): Returns a single data point (image and tokenized prompt).
    """
    def __init__(self, dataset, instance_prompt, tokenizer, size=512):
        self.dataset = dataset
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        # Assuming self.dataset is a list of dictionaries, each containing an "image" key
        image = self.dataset[index]["image"]
        # Apply transforms to the image
        example["instance_images"] = self.transforms(image)
        # Assuming self.instance_prompt is defined somewhere in your class
        # and tokenizer is a CLIPTokenizer instance

        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )["input_ids"]
        return example


def collate_fn(examples):
    """
    Collate function to prepare a batch of data for training.

    Args:
        examples (list): A list of examples where each example is a dictionary
                         containing 'instance_prompt_ids' and 'instance_images'.

    Returns:
        dict: A dictionary containing batched 'input_ids' and 'pixel_values'.
    """
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format)
    pixel_values = pixel_values.float()

    tokenizer = CLIPTokenizer.from_pretrained("./stable_diffusion_models/tokenizer")

    # Tokenize prompts
    batch_encoding = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_attention_mask=True, return_tensors="pt"
    )

    # Extract input_ids and attention_mask from batch_encoding
    input_ids = batch_encoding["input_ids"]
    attention_mask = batch_encoding["attention_mask"]

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
    }
    return batch


def training_function(text_encoder, vae, unet, args, my_bar):
    """
    Function to train the Stable Diffusion model.

    Args:
        text_encoder (CLIPTextModel): The text encoder model.
        vae (AutoencoderKL): The variational autoencoder model.
        unet (UNet2DConditionModel): The U-Net model.
        args (Namespace): A Namespace object containing training arguments and configurations.
        my_bar (streamlit.progress): A Streamlit progress bar to monitor the training progress.

    This function performs the following steps:
        1. Sets up the training environment and configurations.
        2. Loads the data and prepares it for training.
        3. Defines the optimizer and noise scheduler.
        4. Trains the model for the specified number of steps, updating the progress bar.
        5. Saves the trained model pipeline.
    """
    tokenizer = CLIPTokenizer.from_pretrained("./stable_diffusion_models/tokenizer")
    feature_extractor = CLIPFeatureExtractor.from_pretrained("./stable_diffusion_models/feature_extractor")

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    set_seed(args.seed)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        unet.parameters(),  # Only optimize unet
        lr=args.learning_rate,
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    train_dataloader = DataLoader(
        args.train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Move text_encode and vae to gpu
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Only show the progress bar once on each machine
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
            my_bar.progress(int(epoch*100/num_train_epochs), text="Training in progress...")

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using the trained modules and save it
    if accelerator.is_main_process:
        print(f"Loading pipeline and saving to {args.output_dir}...")
        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            steps_offset=1,
        )
        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
            feature_extractor=feature_extractor,
        )
        pipeline.save_pretrained(args.output_dir)


def train_model(product_name, my_bar):
    """
    Train the Stable Diffusion model with images of the specified product.

    Args:
        product_name (str): The name of the product to be used in the training prompt.
        my_bar (streamlit.progress): A Streamlit progress bar to monitor the training progress.

    This function performs the following steps:
        1. Downloads the pretrained model components.
        2. Loads the dataset containing images of the product.
        3. Creates a DreamBoothDataset with the loaded images and the product name.
        4. Sets up training arguments and configurations.
        5. Initializes the model components.
        6. Launches the training process using the specified number of GPUs.
        7. Saves the trained model pipeline.
    """
    download_model(my_bar)
    dataset = load_dataset("imagefolder", data_dir="training_photos/", split='train')
    instance_prompt = f"a photo of a {product_name}"
    learning_rate = 2e-06
    max_train_steps = 200
    load_directory = "./stable_diffusion_models"
    tokenizer = CLIPTokenizer.from_pretrained(f"{load_directory}/tokenizer")
    train_dataset = DreamBoothDataset(dataset, instance_prompt, tokenizer)
    args = Namespace(
        pretrained_model_name_or_path=load_directory,
        resolution=512,  # Reduce this if you want to save some memory
        train_dataset=train_dataset,
        instance_prompt=instance_prompt,
        learning_rate=learning_rate,
        max_train_steps=max_train_steps,
        train_batch_size=1,
        gradient_accumulation_steps=1,  # Increase this if you want to lower memory usage
        max_grad_norm=1.0,
        gradient_checkpointing=True,  # Set this to True to lower the memory usage
        use_8bit_adam=True,  # Use 8bit optimizer from bitsandbytes
        seed=3434554,
        sample_batch_size=2,
        output_dir="./pipeline-folder",  # Where to save the pipeline
    )
    text_encoder = CLIPTextModel.from_pretrained(f"{load_directory}/text_encoder")
    vae = AutoencoderKL.from_pretrained(f"{load_directory}/vae")
    unet = UNet2DConditionModel.from_pretrained(f"{load_directory}/unet")
    num_of_gpus = 1  # CHANGE THIS TO MATCH THE NUMBER OF GPUS YOU HAVE
    notebook_launcher(
        training_function, args=(text_encoder, vae, unet, args, my_bar), num_processes=num_of_gpus
    )
    my_bar.progress(96, text="Saving the model...")
    with torch.no_grad():
        torch.cuda.empty_cache()
