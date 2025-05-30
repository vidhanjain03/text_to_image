# 🎨 Text-to-Image Generator with Stable Diffusion XL

This Streamlit app allows users to generate stunning images from text prompts using **Stable Diffusion XL** via the **Hugging Face Inference API**.


<p align="center"> <img src="https://github.com/vidhanjain03/text_to_image/blob/main/banner.png" alt="App Screenshot" width="700"/> </p>

---

## 🚀 Live Demo

👉 [Try it on Streamlit!](https://vidhan-text2image.streamlit.app/)  



## ✨ Features

- 🔥 Powered by [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- ⚡ Live image generation using Hugging Face Inference API
- 🖼️ Outputs 512x512 quality images
- 🎛️ Simple, responsive interface built with Streamlit
- 🔒 Secure handling of secrets using `st.secrets`

---

## 📦 Dependencies

Make sure you have the following Python packages installed:

```bash
pip install -r requirements.txt
```

If requirements.txt is missing
```bash
pip install streamlit requests pillow python-dotenv
```

## 📁 Project Structure
```
text_to_image/
├── App.py                # Main Streamlit app
├── .streamlit/
│   └── secrets.toml      # For local testing (optional)
├── requirements.txt      # Python dependencies
└── README.md             # You are here
```
