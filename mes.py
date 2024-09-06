import streamlit as st
import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from huggingface_hub import hf_hub_download
import os
import requests
from huggingface_hub import login

# Function to run the first model
def run_model_1(prompt, token):
    try:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            use_auth_token=token  # Use the token here
        )
        pipe.enable_model_cpu_offload()
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(0)
        ).images[0]
        image.save("flux-dev.png")
        st.image("flux-dev.png", caption="Output from Model 1")
    except Exception as e:
        st.error(f"Error running Model 1: {e}")

# Function to run the second model
def run_model_2(prompt):
    try:
        pipe = FluxPipeline.from_pretrained(
            "Shakker-Labs/AWPortrait-FL",
            torch_dtype=torch.bfloat16
        )
        pipe.to("cuda")
        image = pipe(
            prompt,
            num_inference_steps=24,
            guidance_scale=3.5,
            width=768,
            height=1024
        ).images[0]
        image.save("awportrait-fl.png")
        st.image("awportrait-fl.png", caption="Output from Model 2")
    except Exception as e:
        st.error(f"Error running Model 2: {e}")

# Function to run the third model
def run_model_3(prompt, token):
    try:
        base_model_id = "black-forest-labs/FLUX.1-dev"
        repo_name = "ByteDance/Hyper-SD"
        ckpt_name = "Hyper-FLUX.1-dev-8steps-lora.safetensors"
        pipe = FluxPipeline.from_pretrained(
            base_model_id,
            use_auth_token=token  # Use the token here
        )
        pipe.load_lora_weights(hf_hub_download(repo_name, ckpt_name, use_auth_token=token))  # Use the token here
        pipe.fuse_lora(lora_scale=0.125)
        pipe.to("cpu", dtype=torch.float16)
        image = pipe(
            prompt=prompt,
            num_inference_steps=8,
            guidance_scale=3.5
        ).images[0]
        image.save("hyper-flux.png")
        st.image("hyper-flux.png", caption="Output from Model 3")
    except Exception as e:
        st.error(f"Error running Model 3: {e}")

# Function to run the fourth model
def run_model_4(prompt):
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cpu")
        image = pipe(
            prompt,
            negative_prompt="",
            num_inference_steps=28,
            guidance_scale=7.0
        ).images[0]
        image.save("stable-diffusion3.png")
        st.image("stable-diffusion3.png", caption="Output from Model 4")
    except Exception as e:
        st.error(f"Error running Model 4: {e}")

# Streamlit interface
st.title("Image Generation Models Runner")

# Input field for prompt
prompt = st.text_input("Enter your prompt:", "A cat holding a sign that says hello world")

# Input field for Hugging Face token (kept secret from display)
token = st.text_input("Enter your Hugging Face token:", type="password")

# Button to run models sequentially
if st.button("Run Models Sequentially"):
    if prompt and token:  # Check if the prompt and token are not empty
        with st.spinner("Running Model 1..."):
            run_model_1(prompt, token)
            st.success("Model 1 completed.")

        with st.spinner("Running Model 2..."):
            run_model_2(prompt)  # No token needed for this one
            st.success("Model 2 completed.")

        with st.spinner("Running Model 3..."):
            run_model_3(prompt, token)
            st.success("Model 3 completed.")

        with st.spinner("Running Model 4..."):
            run_model_4(prompt)  # No token needed for this one
            st.success("Model 4 completed.")
    else:
        st.error("Please enter both a prompt and your Hugging Face token to proceed.")
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