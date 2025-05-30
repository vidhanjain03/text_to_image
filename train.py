## This code is compiled models_utils.py and MyApp.py

from diffusers import DiffusionPipeline
import torch
import streamlit as st
from PIL import Image
import io

import os
os.environ["PYTORCH_DISABLE_MPS_FALLBACK"] = "1"
os.environ["STREAMLIT_WATCH_DIRECTORIES_USE_POLLING"] = "true"

# Load Stable Diffusion XL Base1.0
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# Optional CPU offloading to save some GPU Memory
pipe.enable_model_cpu_offload()

# Loading Trained LoRA Weights
pipe.load_lora_weights("AdamLucek/sdxl-base-1.0-oldbookillustrations-lora")

def generate_image(prompt: str) -> Image.Image:
    # Invoke pipeline to generate image
    image = pipe(
        prompt = prompt,
        num_inference_steps=50,
        height=1024,
        width=1024,
        guidance_scale=7.0).images[0]
    #img = Image.new('RGB',(512,512),color='white')
    return image

st.set_page_config(page_title="Text to Image Generation", layout="centered")

st.title("Text to Image Generation")
st.markdown("Enter Prompt and generate and Image using Stable Diffusion XL Base 1.0")

prompt = st.text_input("Enter your prompt here")

if st.button("Generate Image"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Generating Image"):
            image = generate_image(prompt)
            st.image(image)