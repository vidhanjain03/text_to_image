import requests
import io
import time
from PIL import Image
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Hugging Face API setup
hf_token = st.secrets["HUGGINGFACE_TOKEN"]
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
#headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"} now.env file is not needed
headers = {"Authorization": f"Bearer {hf_token}"}

# Function to call Hugging Face inference API
def query(payload, retries=5):
    for attempt in range(retries):
        print(f"Attempt {attempt+1} of {retries}")
        response = requests.post(API_URL, headers=headers, json=payload)
        content_type = response.headers.get("content-type", "")

        if "image" in content_type:
            print("✅ Image successfully generated!")
            return response.content
        else:
            try:
                error_info = response.json()
                print("⚠️ API response:", error_info)
                if "estimated_time" in error_info:
                    wait_time = int(error_info["estimated_time"]) + 5
                    print(f"⏳ Model is loading. Waiting for {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("❌ Non-retryable error. Exiting.")
                    return None
            except Exception as e:
                print("❌ Error parsing JSON response:", e)
                return None
    print("❌ All retries exhausted. No image generated.")
    return None

# Streamlit page configuration
st.set_page_config(page_title="Text to Image Generator", layout="centered")

# Create three columns: spacer | app content | spacer
spacer_left, content_col, spacer_right = st.columns([1, 4, 1])  # Adjust ratios to control margins

with content_col:
    st.title("🎨 Text to Image Generator with Stable Diffusion XL")

    # Split the main content column into two: left for prompt, right for image
    left_col, right_col = st.columns([1, 2])  # Adjust width ratios here too

    with left_col:
        st.header("📝 Prompt")
        prompt = st.text_area("Describe what you want to generate:", height=150)
        generate_button = st.button("✨ Generate Image")

    with right_col:
        if generate_button:
            if not prompt.strip():
                st.warning("Please enter a prompt to generate an image.")
            else:
                with st.spinner("🖼️ Generating image..."):
                    image_bytes = query({"inputs": prompt})
                    if image_bytes:
                        image = Image.open(io.BytesIO(image_bytes))
                        st.image(image, caption="Generated Image", width=512)
                    else:
                        st.error("Failed to generate image. Please try again.")

