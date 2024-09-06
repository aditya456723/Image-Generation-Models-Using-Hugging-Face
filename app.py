import streamlit as st
import torch
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from huggingface_hub import hf_hub_download
import os
import requests
from huggingface_hub import login
import io
from PIL import Image
from IPython.display import display
import requests

def run_model_1(prompt, token):
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    headers = {"Authorization": "Bearer "+token}
    
    # Define the payload with the prompt
    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes))
    image.save(prompt+".png")
    st.image(prompt+".png", caption="Output from Model 1")
    return  
    
# Function to run the second model
def run_model_2(prompt, token):
    API_URL = "https://api-inference.huggingface.co/models/ByteDance/Hyper-SD"
    headers = {"Authorization": "Bearer "+token}
    
    # Define the payload with the prompt
    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    image_bytes = response.content
    image = Image.open(io.BytesIO(image_bytes))
    image.save(prompt+"1.png")
    st.image(prompt+"1.png", caption="Output from Model 2")
    return 



# Function to run the fourth model
def run_model_3(prompt, token):
    API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    headers = {"Authorization": f"Bearer {token}"}
    
    # Define the payload with the prompt
    payload = {"inputs": prompt}

    try:
        # Make the API request
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Attempt to open the image
            image_bytes = response.content
            try:
                image = Image.open(io.BytesIO(image_bytes))
                image.save(prompt+"3.png")
                return st.image(prompt+"3.png", caption="Output from Model 3")  # Display the image
            except IOError as e:
                print(f"Error opening image: {e}")
        else:
            print(f"Failed to retrieve image. HTTP Status code: {response.status_code}")
            print("Response content:", response.content)
    except requests.RequestException as e:
        print(f"Request failed: {e}")     
    
    return 
        
   
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
            run_model_3(prompt, token)
            st.success("Model 1 completed.")

        with st.spinner("Running Model 2..."):
            run_model_2(prompt, token)  
            st.success("Model 2 completed.")



        with st.spinner("Running Model 3..."):
            run_model_1(prompt, token)  
            st.success("Model 3 completed.")
    else:
        st.error("Please enter both a prompt and your Hugging Face token to proceed.")
