import base64
import time
import urllib

import openai
import requests
import streamlit as st

#from streamlit_lottie import st_lottie
#from streamlit_lottie import st_lottie_spinner
from PIL import Image
from streamlit_chat import message

import config as p
from overlay import *
from real_image import *
from config import *
from cation import *
import subprocess

st.set_page_config(layout="wide")
col1, col2 = st.columns([1,3])
with col1:
    st.image('Nestle_purina_logo.png')
with col2:
    original_title = '<p style="Arial"; Color:green; font-size : 30px;><h2><div style="text-align: left">NESTLE PURINA CONTENT GENERATION </div></h2></p>'
    st.markdown(original_title, unsafe_allow_html=True)

openai.api_key= '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'

# Session States:
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "system", "content": "You are a helpful assistant."}]


cols = st.columns(4)
with cols[0]:
    option1 = st.selectbox(
        "Person", ("Man", "Woman", "Boy", "Girl")
    )
    st.write("You selected:", option1)
    st.write("---")
    st.write("##")
with cols[1]:
    option2 = st.selectbox(
        "Ethnicity", ("Caucasian", "African-American")
    )
    st.write("You selected:", option2)
    st.write("---")
    st.write("##")
with cols[2]:
    option3 = st.selectbox(
        "Animal", ("Husky", "Labrador", "Golden-Retreiver", "Persian Cat", "American Shorthair Cat", "British Shorthair Cat")
    )
    st.write("You selected:", option3)
    st.write("---")
    st.write("##")

option5 = 'Theme'
with cols[3]:
    option5 = st.selectbox("Theme", ("Living Room", "Park", "Pet Clinic"))
    st.write("You selected:", option5)
    st.write("---")
    st.write("##")

prompt = f"Provide a description of an image showing {option1} of {option2} ethnicity with a {option3} in a {option5}  in 10 words."

#if option5 == "Pet Clinic":
#    prompt  =  f"Provide a description of an image showing a doctor of {option2} ethnicity with a {option3} in a {option5}  in 10 words."

generate = st.button('GENERATE')
captions = ""
if generate:
    try:
      st.session_state['messages'] = [
          {"role": "system", "content": "You are a helpful assistant."}
      ]
      st.session_state['messages'].append({"role": "user", "content": prompt})
      completion = openai.ChatCompletion.create(
          model='gpt-3.5-turbo',
          messages=st.session_state['messages']
      )
      response = completion.choices[0].message.content
      image_prompt = response
      captions = response
      print(image_prompt)
    except:
      response = f"A {option2} {option1} is with a {option3} {option5}."
      image_prompt = response
      captions = response

    #Getting images
    #print(option1)
    imgs = inference(prompt=image_prompt, negative_prompt=neg_prompt, num_samples=5, height=512, width=512, num_inference_steps=60, guidance_scale=7, person = option1, ethinicity = option2 , animal = option3, theme = option5)
    for i in range(len(imgs)):
        imgs[i].save(f"{image_folder}/{i+1}.png")

    command = "python /home/suchitra/CodeFormer/inference_codeformer.py -w 0.7 --input_path /home/suchitra/kellogs/Outputs/Image/ --bg_upsampler realesrgan --face_upsample --output_path /home/suchitra/kellogs/Outputs/enhanced/"

    # Run the command
    subprocess.run(command, shell=True)

    img_paths = os.listdir('/home/suchitra/kellogs/Outputs/enhanced/final_results/')

    for img in img_paths:
      add_cation(str(captions), img, '/home/suchitra/kellogs/Outputs/enhanced/final_results/', '/home/suchitra/kellogs/Outputs/captioned/')

    if len(os.listdir(out_path))>0:
        for image in os.listdir(out_path):
            image_pth = os.path.join(out_path,image)
            st.image(image_pth,caption = captions, width = 300)
            with open(image_pth, "rb") as file:
                btn = st.download_button(
                        label="Download image",
                        data=file,
                        file_name=image_pth,
                        mime="image/png"
                      )
