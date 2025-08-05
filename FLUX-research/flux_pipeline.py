import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer, CLIPTextModel, CLIPTokenizer
from custom_diffusers import FluxTransformer2DModel, FluxPipeline
from custom_diffusers.models.autoencoders import AutoencoderKL
import numpy as np

class ManualFluxRunner:
    def __init__(self, model_path="FLUX.1-dev", device="cpu"):
        self.device = device
        self.model_path = model_path
        
        print("Loading models...")
        
        # Load T5 text encoder (primary text encoder)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(
            model_path, subfolder="tokenizer_2", legacy=False
        )
        self.t5_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder_2", torch_dtype=torch.bfloat16
        ).to(device)
        
        # Load CLIP text encoder (secondary text encoder) 
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(
            model_path, subfolder="tokenizer"
        )
        self.clip_encoder = CLIPTextModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
        ).to(device)
        
        # Load the main transformer model
        self.transformer = FluxTransformer2DModel.from_pretrained(
            model_path, subfolder="transformer", torch_dtype=torch.bfloat16
        ).to(device)
        
        # Load VAE for encoding/decoding images
        self.vae = AutoencoderKL.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.bfloat16
        ).to(device)
        
        # Get scheduler from pipeline for noise scheduling
        temp_pipeline = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.scheduler = temp_pipeline.scheduler
        del temp_pipeline  # Free memory
        
        print("All models loaded successfully!")

    def encode_text_t5(self, prompt, max_length=512):
        """Encode text using T5 encoder and return embeddings"""
        print(f"Encoding with T5: '{prompt}'")
        
        # Tokenize
        text_inputs = self.t5_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        
        # Encode
        with torch.no_grad():
            text_embeddings = self.t5_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
        
        print(f"T5 embeddings shape: {text_embeddings.shape}")
        print(f"T5 embeddings dtype: {text_embeddings.dtype}")
        print(f"T5 embeddings sample values: {text_embeddings[0, :3, :5]}")
        
        return text_embeddings, attention_mask

    def encode_text_clip(self, prompt, max_length=77):
        """Encode text using CLIP encoder and return embeddings"""
        print(f"Encoding with CLIP: '{prompt}'")
        
        # Tokenize
        text_inputs = self.clip_tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        
        # Encode
        with torch.no_grad():
            text_embeddings = self.clip_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
        
        print(f"CLIP embeddings shape: {text_embeddings.shape}")
        print(f"CLIP embeddings dtype: {text_embeddings.dtype}")
        print(f"CLIP embeddings sample values: {text_embeddings[0, :3, :5]}")
        
        return text_embeddings, attention_mask

    def prepare_latents(self, batch_size=1, height=1024, width=1024, generator=None):
        """Initialize random latents for generation"""
        # Flux uses 8x downsampling for VAE, and latent channels = 16
        latent_height = height // 8  # VAE downsampling
        latent_width = width // 8
        latent_channels = 16  # Flux latent channels
        
        shape = (batch_size, latent_channels, latent_height, latent_width)
        
        if generator is None:
            generator = torch.Generator(device=self.device)
        
        # Flux uses standard normal distribution for initial latents
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=torch.bfloat16)
        
        print(f"Initial latents shape: {latents.shape}")
        print(f"Initial latents dtype: {latents.dtype}")
        print(f"Latents std: {latents.std():.4f}, mean: {latents.mean():.4f}")
        
        return latents

    def run_denoising_step(self, latents, t5_embeds, t5_attention_mask, clip_embeds, 
                          clip_attention_mask, timestep, guidance_scale=3.5, 
                          uncond_t5_embeds=None, uncond_clip_embeds=None):
        """Run a single denoising step manually"""
        print(f"Denoising step for timestep: {timestep}")
        
        # Prepare timestep - Flux uses float timesteps
        timestep_tensor = torch.tensor([timestep], device=self.device, dtype=torch.float32)
        
        # For classifier-free guidance, we need to prepare conditional and unconditional embeddings
        if guidance_scale > 1.0:
            # Use cached unconditional embeddings or create them
            if uncond_t5_embeds is None or uncond_clip_embeds is None:
                uncond_t5_embeds, uncond_t5_mask = self.encode_text_t5("")
                uncond_clip_embeds, uncond_clip_mask = self.encode_text_clip("")
            else:
                uncond_t5_mask = torch.ones(uncond_t5_embeds.shape[:2], device=self.device)
                uncond_clip_mask = torch.ones(uncond_clip_embeds.shape[:2], device=self.device)
            
            # Concatenate conditional and unconditional
            t5_embeds_combined = torch.cat([uncond_t5_embeds, t5_embeds])
            t5_mask_combined = torch.cat([uncond_t5_mask, t5_attention_mask])
            clip_embeds_combined = torch.cat([uncond_clip_embeds, clip_embeds])
            
            # Duplicate latents for conditional and unconditional
            latents_input = torch.cat([latents, latents])
            timestep_tensor = timestep_tensor.repeat(2)
        else:
            t5_embeds_combined = t5_embeds
            t5_mask_combined = t5_attention_mask
            clip_embeds_combined = clip_embeds
            latents_input = latents
        
        # Pool CLIP embeddings - Flux expects pooled text embeddings
        pooled_clip = clip_embeds_combined.mean(dim=1)
        
        print(f"Latents input shape: {latents_input.shape}")
        print(f"T5 embeddings shape: {t5_embeds_combined.shape}")
        print(f"Pooled CLIP shape: {pooled_clip.shape}")
        print(f"Timestep: {timestep_tensor}")
        
        # Run transformer
        with torch.no_grad():
            noise_pred = self.transformer(
                hidden_states=latents_input,
                encoder_hidden_states=t5_embeds_combined,
                # encoder_attention_mask=t5_mask_combined,
                pooled_projections=pooled_clip,
                timestep=timestep_tensor,
                return_dict=False
            )[0]
        
        # Apply classifier-free guidance
        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        print(f"Noise prediction shape: {noise_pred.shape}")
        print(f"Noise prediction std: {noise_pred.std():.4f}")
        
        return noise_pred

    def generate_step_by_step(self, prompt, num_inference_steps=20, guidance_scale=7.5, 
                             height=1024, width=1024, seed=None):
        """Full generation with manual control over each step"""
        print(f"Starting generation: '{prompt}'")
        print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, Size: {height}x{width}")
        
        # Set up generator
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            print(f"Using seed: {seed}")
        
        # Step 1: Encode text with both encoders
        print("\n=== STEP 1: Text Encoding ===")
        t5_embeds, t5_mask = self.encode_text_t5(prompt)
        clip_embeds, clip_mask = self.encode_text_clip(prompt)
        
        # Step 2: Prepare initial latents
        print("\n=== STEP 2: Latent Initialization ===")
        latents = self.prepare_latents(1, height, width, generator)
        
        # Step 3: Set up scheduler
        print("\n=== STEP 3: Scheduler Setup ===")
        self.scheduler.set_timesteps(num_inference_steps, device=self.device, mu=0.0)
        timesteps = self.scheduler.timesteps
        print(f"Timesteps: {timesteps}")
        
        # Step 4: Denoising loop
        print("\n=== STEP 4: Denoising Loop ===")
        for i, timestep in enumerate(timesteps):
            print(f"\n--- Denoising step {i+1}/{num_inference_steps} ---")
            
            # Get noise prediction
            noise_pred = self.run_denoising_step(
                latents, t5_embeds, t5_mask, clip_embeds, clip_mask, 
                timestep, guidance_scale
            )
            
            # Update latents using scheduler
            latents = self.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
            
            print(f"Updated latents std: {latents.std():.4f}")
        
        # Step 5: Decode latents to image
        print("\n=== STEP 5: VAE Decoding ===")
        with torch.no_grad():
            # Scale latents back
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]
        
        # Convert to numpy and denormalize
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        
        print(f"Final image shape: {image.shape}")
        print("Generation complete!")
        
        return image[0]  # Return first image

# Usage example
def main():
    # Initialize the manual runner
    runner = ManualFluxRunner()
    
    # Generate an image step by step
    prompt = "A majestic dragon flying over a medieval castle at sunset"
    
    image = runner.generate_step_by_step(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=3.5,  # Flux typically uses lower guidance scale
        height=1024,
        width=1024,
        seed=42
    )
    
    # Save the result
    from PIL import Image
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    pil_image.save("flux_manual_output.png")
    print("Image saved as flux_manual_output.png")

if __name__ == "__main__":
    main()