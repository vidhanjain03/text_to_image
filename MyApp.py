import streamlit as st
from models_utils import load_model, generate_image
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["STREAMLIT_WATCH_DIRECTORIES_USE_POLLING"] = "true"
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"


# @st.cache_resource
def get_pipeline():
    return load_model()

pipe = get_pipeline()

st.set_page_config(page_title="Text to Image Generation", layout="centered")

st.title("Text to Image Generation")
st.markdown("Enter Prompt and generate an Image using Stable Diffusion XL Base 1.0")

prompt = st.text_input("Enter your prompt here")

if st.button("Generate Image"):
    if prompt.strip() == "":
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Generating Image"):
            try:
                image = generate_image(pipe, prompt)
                st.image(image)
            except Exception as e:
                st.error(f"Error generating image: {str(e)}")