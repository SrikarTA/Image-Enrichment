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
import streamlit as st
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
#if 'past' not in st.session_state:
#    st.session_state['past'] = []
#if 'messages' not in st.session_state:
#    st.session_state['messages'] = [{"role": "system", "content": "You are a helpful assistant."}]
    


# Create the first dropdown for the main category
selected_option = st.selectbox(
    "Select an Theme",
    (
        "Seeking Authenticity",
        "Social Consciousness",
        "Healthy Lifestyle",
        "Feeding Experience",
        "Utilitarian and Nutrition"
    )
)

if selected_option == "Seeking Authenticity":
    # Display the second dropdown for "Seeking Authenticity"
    sub_option = st.selectbox(
        "Select a Sub-Option",
        (
            "Farmland",
            "Food Bowl"
        )
    )
    st.write("You selected:", sub_option)
    
    if sub_option == "Farmland":
      prompt = f"Provide a description of an image of farmland with tractor in 10 words."
    elif sub_option == "Poultry":
      prompt = f"Provide a description of an image of poultry farm of chicken and turkey in 10 words."
    else :
      prompt = f"Provide a description of an image of food bowl containg peas, carrot and salmon in 10 words."
    
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
        response = f"A {sub_option} with all essential elements."
        image_prompt = response
        captions = response
        
      #Getting images
      #print(option1)
      imgs = inference1(prompt=image_prompt, negative_prompt=neg_prompt, num_samples=5, height=512, width=512, num_inference_steps=60, guidance_scale=7, theme = sub_option, )
      for i in range(len(imgs)):
          imgs[i].save(f"{image_folder}/{i+1}.png")
    
      #Overlay product
    #    overlay_img(background_img_folder,option5)
    
    
    # Enhancing Image
  
      
      command = "python /home/suchitra/CodeFormer/inference_codeformer.py -w 0.7 --input_path /home/suchitra/kellogs/Outputs/Image/ --bg_upsampler realesrgan --face_upsample --output_path /home/suchitra/kellogs/Outputs/enhanced/"
      
      # Run the command
      subprocess.run(command, shell=True)
      
      # Captioning_image
      img_paths = os.listdir('/home/suchitra/kellogs/Outputs/enhanced/final_results/')
      
      for img in img_paths:
        add_cation(str(captions), img, '/home/suchitra/kellogs/Outputs/enhanced/final_results/', '/home/suchitra/kellogs/Outputs/captioned/')
        
      #  Adding Logos to image 
      img_paths = os.listdir('/home/suchitra/kellogs/Outputs/captioned/')
      
      if sub_option == "Farmland":
        for img in img_paths:
          add_logos(img,'/home/suchitra/kellogs/Outputs/captioned/','/home/suchitra/kellogs/theme_logos/sa_farmland/', out_path)
          
      elif sub_option == "Poultry":
        for img in img_paths:
          add_logos(img,'/home/suchitra/kellogs/Outputs/captioned/','/home/suchitra/kellogs/theme_logos/sa_poultry- Copy/', out_path)
          
      else :
        for img in img_paths:
          add_logos(img,'/home/suchitra/kellogs/Outputs/captioned/','/home/suchitra/kellogs/theme_logos/sa_food_bowl/', out_path)

elif selected_option == "Social Consciousness":
    # Create a horizontal layout for "Social Conciousness" dropdowns
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
      option5 = st.selectbox("Theme", ("Living Room", "Park"))
      st.write("You selected:", option5)
      st.write("---")
      st.write("##")
    
    age_range = '30s' if option1 in ['Man', 'Woman'] else 'early teens'
    
    prompt = f"Provide a description of an image showing {option1} in {age_range} of {option2} ethnicity with a {option3} in {option5} in 10 words."
    
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
      imgs = inference(prompt=image_prompt, negative_prompt=neg_prompt, num_samples=5, height=512, width=512, num_inference_steps=60, guidance_scale=7, person = option1, ethinicity = option2 , animal = option3, theme = option5, selected_option = selected_option)
      for i in range(len(imgs)):
          imgs[i].save(f"{image_folder}/{i+1}.png")
    
      #Overlay product
    #    overlay_img(background_img_folder,option5)
    
      command = "python /home/suchitra/CodeFormer/inference_codeformer.py -w 0.7 --input_path /home/suchitra/kellogs/Outputs/Image/ --bg_upsampler realesrgan --face_upsample --output_path /home/suchitra/kellogs/Outputs/enhanced/"
      
      # Run the command
      subprocess.run(command, shell=True)
      
      img_paths = os.listdir('/home/suchitra/kellogs/Outputs/enhanced/final_results/')
      
      for img in img_paths:
        add_cation(str(captions), img, '/home/suchitra/kellogs/Outputs/enhanced/final_results/', '/home/suchitra/kellogs/Outputs/captioned/')
      
      #  Adding Logos to image 
      img_paths = os.listdir('/home/suchitra/kellogs/Outputs/captioned/')
      
      for img in img_paths:
        add_logos(img,'/home/suchitra/kellogs/Outputs/captioned/','/home/suchitra/kellogs/theme_logos/sc/', out_path)

        
elif selected_option == "Healthy Lifestyle":
    # Create a horizontal layout for "Feeding Experience" dropdowns
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
          "Animal", ("Labrador", "Golden-Retreiver", "Husky")
      )
      st.write("You selected:", option3)
      st.write("---")
      st.write("##")
      
    option5 = 'Theme'
    with cols[3]:
      option5 = st.selectbox("Theme", ("Camping", "Trekking", "Park Walking"))
      st.write("You selected:", option5)
      st.write("---")
      st.write("##")
      
    age_range = '30s' if option1 in ['Man', 'Woman'] else 'early teens'
    
    if option5 == "Trekking":
      prompt = f"Provide a description of an image showing {option1} in {age_range} of {option2} ethnicity sitting on rocks in mountains with a {option3} in 10 words."
    elif option5 == "Park walking":
      prompt = f"Provide a description of an image showing {option1} in {age_range} of {option2} ethnicity standing in a park with a {option3} in 10 words."
    else:
      prompt = f"Provide a description of an image showing {option1} in {age_range} of {option2} ethnicity with a {option3} while {option5} in 10 words."
    
    
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
        response = f"A {option1} of {option2} ethnicity waith a {option3} while {option5}."
        image_prompt = response
        captions = response
        
      #Getting images
      #print(option1)
      imgs = inference(prompt=image_prompt, negative_prompt=neg_prompt, num_samples=5, height=512, width=512, num_inference_steps=60, guidance_scale=7, person = option1, ethinicity = option2 , animal = option3, theme = option5, selected_option = selected_option)
      for i in range(len(imgs)):
          imgs[i].save(f"{image_folder}/{i+1}.png")
    
      #Overlay product
    #    overlay_img(background_img_folder,option5)
    
      command = "python /home/suchitra/CodeFormer/inference_codeformer.py -w 0.7 --input_path /home/suchitra/kellogs/Outputs/Image/ --bg_upsampler realesrgan --face_upsample --output_path /home/suchitra/kellogs/Outputs/enhanced/"
      
      # Run the command
      subprocess.run(command, shell=True)
      
      img_paths = os.listdir('/home/suchitra/kellogs/Outputs/enhanced/final_results/')
      
      for img in img_paths:
        add_cation(str(captions), img, '/home/suchitra/kellogs/Outputs/enhanced/final_results/', '/home/suchitra/kellogs/Outputs/captioned/')
      
      #  Adding Logos to image 
      img_paths = os.listdir('/home/suchitra/kellogs/Outputs/captioned/')
      
      for img in img_paths:
        add_logos(img,'/home/suchitra/kellogs/Outputs/captioned/','/home/suchitra/kellogs/theme_logos/healthy_lifestyle/', out_path)

elif selected_option == "Feeding Experience":
    # Create a horizontal layout for "Feeding Experience" dropdowns
    cols = st.columns(2)
    #with cols[0]:
    #  option1 = st.selectbox(
    #      "Person", ("Man", "Woman", "Boy", "Girl")
    #  )
    #  st.write("You selected:", option1)
    #  st.write("---")
    #  st.write("##")
    #with cols[1]:
    #  option2 = st.selectbox(
    #      "Ethnicity", ("Caucasian", "African-American")
    #  )
    #  st.write("You selected:", option2)
    #  st.write("---")
    #  st.write("##")
    with cols[0]:
      option3 = st.selectbox(
          "Animal", ("Husky", "Labrador", "Golden-Retreiver", "Persian Cat", "American Shorthair Cat", "British Shorthair Cat")
      )
      st.write("You selected:", option3)
      st.write("---")
      st.write("##")
      
    option5 = 'Theme'
    with cols[1]:
      option5 = st.selectbox("Theme", ("Living Room", "Park"))
      st.write("You selected:", option5)
      st.write("---")
      st.write("##")
    
    #age_range = '30s' if option1 in ['Man', 'Woman'] else 'early teens'
    
    prompt = f"Provide a description of an image showing {option3} eating food from a food bowl in {option5}in 10 words."
    
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
        response = f"A {option2} {option1} is feeding food to a {option3} {option5}."
        image_prompt = response
        captions = response
      
      #Getting images
      #print(option1)
      imgs = inference(prompt=prompt, negative_prompt=neg_prompt, num_samples=5, height=512, width=512, num_inference_steps=60, guidance_scale=7, person = "", ethinicity = "" , animal = option3, theme = option5, selected_option = selected_option)
      for i in range(len(imgs)):
          imgs[i].save(f"{image_folder}/{i+1}.png")
    
      #Overlay product
    #    overlay_img(background_img_folder,option5)
    
      command = "python /home/suchitra/CodeFormer/inference_codeformer.py -w 0.7 --input_path /home/suchitra/kellogs/Outputs/Image/ --bg_upsampler realesrgan --face_upsample --output_path /home/suchitra/kellogs/Outputs/enhanced/"
      
      # Run the command
      subprocess.run(command, shell=True)
      
      img_paths = os.listdir('/home/suchitra/kellogs/Outputs/enhanced/final_results/')
      
      for img in img_paths:
        add_cation(str(captions), img, '/home/suchitra/kellogs/Outputs/enhanced/final_results/', '/home/suchitra/kellogs/Outputs/captioned/')
      
      #  Adding Logos to image 
      img_paths = os.listdir('/home/suchitra/kellogs/Outputs/captioned/')
      
      for img in img_paths:
        add_logos(img,'/home/suchitra/kellogs/Outputs/captioned/','/home/suchitra/kellogs/theme_logos/feeding_experience/', out_path)
      
                    
elif  selected_option == "Utilitarian and Nutrition":
    
    cols = st.columns(4)
    with cols[0]:
      option1 = st.selectbox(
          "Person", ("Male", "Female", "Boy", "Girl")
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
      option5 = st.selectbox("Theme", ("Living Room", "Pet Clinic"))
      st.write("You selected:", option5)
      st.write("---")
      st.write("##")
    
    age_range = '30s' if option1 in ['Male', 'Female'] else 'early teens'
    
    prompt = f"Provide a description of an image showing {option1} in {age_range} of {option2} ethnicity with a {option3} in {option5}  in 10 words."
    
    if option5 == "Pet Clinic":
        prompt  =  f"Provide a description of an image showing a {option1} doctor of {option2} ethnicity with a {option3} in a {option5}  in 10 words."
      
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
      imgs = inference(prompt=image_prompt, negative_prompt=neg_prompt, num_samples=5, height=512, width=512, num_inference_steps=60, guidance_scale=7, person = option1, ethinicity = option2 , animal = option3, theme = option5, selected_option = selected_option)
      for i in range(len(imgs)):
          imgs[i].save(f"{image_folder}/{i+1}.png")
    
      #Overlay product
    #    overlay_img(background_img_folder,option5)
    
      command = "python /home/suchitra/CodeFormer/inference_codeformer.py -w 0.7 --input_path /home/suchitra/kellogs/Outputs/Image/ --bg_upsampler realesrgan --face_upsample --output_path /home/suchitra/kellogs/Outputs/enhanced/"
      
      # Run the command
      subprocess.run(command, shell=True)
      
      img_paths = os.listdir('/home/suchitra/kellogs/Outputs/enhanced/final_results/')
      
      for img in img_paths:
        add_cation(str(captions), img, '/home/suchitra/kellogs/Outputs/enhanced/final_results/', '/home/suchitra/kellogs/Outputs/captioned/')
      
      #  Adding Logos to image 
      img_paths = os.listdir('/home/suchitra/kellogs/Outputs/captioned/')
      
      for img in img_paths:
        add_logos(img,'/home/suchitra/kellogs/Outputs/captioned/','/home/suchitra/kellogs/theme_logos/u_n/', out_path)
      
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
                

  