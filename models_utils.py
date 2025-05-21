from diffusers import DiffusionPipeline
import torch
from PIL import Image

# Load Stable Diffusion XL Base1.0

def check_offloaded_devices(pipe, verbose=True):
    """Check and print the device placement of model components' parameters."""
    devices = {}
    for name, component in pipe.components.items():
        if isinstance(component, torch.nn.Module):
            param_device = next(component.parameters()).device
            devices[name] = param_device
            if verbose:
                print(f"{name} device: {param_device}")
    return devices


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure a GPU is available or modify the code to use CPU.")

    # Load model on CPU to ensure weights are fully initialized
    print("Loading model on CPU...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    
    # Verify weights are loaded (no meta tensors)
    print("Verifying model weights...")
    for name, component in pipe.components.items():
        if isinstance(component, torch.nn.Module):
            try:
                param = next(component.parameters())
                if param.is_meta:
                    raise RuntimeError(f"Meta tensor detected in {name}")
                print(f"{name} weights loaded on {param.device}")
            except StopIteration:
                print(f"{name} has no parameters")

    # Move to CUDA for inference
    print("Moving model to CUDA...")
    pipe = pipe.to("cuda")
    
    # Optionally load LoRA weights (commented out to avoid meta tensor issues)
    print("Loading LoRA weights...")
    pipe.load_lora_weights("AdamLucek/sdxl-base-1.0-oldbookillustrations-lora")
    
    # Enable CPU offloading
    print("Enabling CPU offloading...")
    pipe.enable_model_cpu_offload()
    
    # Check device placement after offloading
    print("Checking device placement after offloading...")
    check_offloaded_devices(pipe)

    return pipe

def generate_image(pipe, prompt: str) -> Image.Image:
    with torch.no_grad():
        print("Generating image...")
        # Check devices before inference
        check_offloaded_devices(pipe)
        image = pipe(
            prompt=prompt,
            num_inference_steps=25,
            height=512,
            width=512,
            guidance_scale=7.0
        ).images[0]
        # Check devices after inference
        print("Checking device placement after inference...")
        check_offloaded_devices(pipe)
    return image