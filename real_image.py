import cv2
import torch
from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from matplotlib import pyplot as plt
from PIL import Image
from torch import autocast
from config import *

g_cpu = torch.Generator(device='cuda')
g_cpu= g_cpu.manual_seed(-1)

device = "cuda"
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"



# def generate_images(prompt):
#     g_cpu = torch.Generator(device='cuda')
#     g_cpu=g_cpu.manual_seed(777)
#     neg_prompt ="fake, ugly ,disfigured, poorly drawn face, cartoonish, poorly drawn fingers, ugly, disfigured, deformed, surreal"
#     images = pipe(prompt,negative_prompt=neg_prompt, guidance_scale =30, height=512, width=512, num_inference_steps=100,generator = g_cpu)
#     images.images[0].show()

def inference(prompt, negative_prompt=neg_prompt, num_samples=5, height=512, width=512, num_inference_steps=60, guidance_scale=7, person = 'Man', ethinicity = 'African American' , animal = 'Husky', theme = 'Living Room', selected_option = 'Feeding Experience'):
    #g_cpu = torch.Generator(device='cuda')
    #print(person)
    if (animal == 'Persian Cat'):
      pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16)
#     pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/easynegative.safetensors", token="easynegative")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/FastNegativeV2.pt", token="FastNegativeV2")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/bad_prompt_version2.pt", token="bad_prompt_version2")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/negative_hand-neg.pt", token="negative_hand-neg")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/pxrc-3200.pt", token="pxrc-3200")
      pipe.load_lora_weights("/home/suchitra/kellogs/embeddings/LORA/", weight_name="persian_cat.safetensors")
      
#      pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
      pipe.scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True,
                                                    algorithm_type="sde-dpmsolver++"
                                                    )
      pipe = pipe.to(device)
      # pipe.enable_attention_slicing()
      if selected_option == 'Feeding Experience':
        prompt = prompt + 'proportionate legs'
      prompt = prompt.replace('Persian Cat', '((Persian Cat))') + ", single cat, (high detailed face:1.2),(masterpiece, finely detailed beautiful eyes: 1.2), dslr, 8k, 4k, ultrarealistic, realistic, natural skin, textured skin"
      
      if selected_option == 'Feeding Experience':
        negative_prompt = "pxrc-3200" + ",dark photos, ((no clothes)), FastNegativeV2, bad_prompt_version2, negative_hand-neg, big bowl, short legs, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"
        
        
        
      negative_prompt = "pxrc-3200" + ",dark photos, ((no clothes)), FastNegativeV2, bad_prompt_version2, negative_hand-neg, easynegative, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"

      with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
                prompt, height=int(height), width=int(width),
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cpu
            ).images
    elif(animal == 'American Shorthair Cat'):
    
      pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16)
#      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/easynegative.safetensors", token="easynegative")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/FastNegativeV2.pt", token="FastNegativeV2")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/bad_prompt_version2.pt", token="bad_prompt_version2")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/negative_hand-neg.pt", token="negative_hand-neg")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/amershrtsq-1900.pt", token="amershrtsq-1900")
      pipe.load_lora_weights("/home/suchitra/kellogs/embeddings/LORA/", weight_name="american_short_hair_cat.safetensors")
#      pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
      pipe.scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True,
                                                    algorithm_type="sde-dpmsolver++"
                                                    )
      pipe = pipe.to(device)
      # pipe.enable_attention_slicing()
      if selected_option == 'Feeding Experience':
        prompt = prompt + 'proportionate legs'
      prompt = prompt.replace('American Shorthair Cat', '((American Shorthair Cat))') + "<lora:american_short_hair_cat:1.2>" + ", single cat,(high detailed face:1.2),(masterpiece, finely detailed beautiful eyes: 1.2), dslr, 8k, 4k, ultrarealistic, realistic, natural skin, textured skin"
      
      
      if selected_option == 'Feeding Experience':
        negative_prompt = "amershrtsq-1900" + ", dark photos, ((no clothes)), FastNegativeV2, bad_prompt_version2, negative_hand-neg, big bowls, short legs, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"
      
      negative_prompt = "amershrtsq-1900" + ", dark photos, ((no clothes)), FastNegativeV2, bad_prompt_version2, negative_hand-neg, easynegative, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"

      with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
                prompt, height=int(height), width=int(width),
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cpu
            ).images
            
    elif(animal == 'British Shorthair Cat'):
    #amnmdl_brtsh-2850, FastNegativeV2, bad_prompt_version2, negative_hand-neg,
    #FastNegativeV2,  bad_prompt_version2, negative_hand-neg,  amnmdl_brtsh-2850, bad anatomy, bad cat eyes
    #british_short_hair_cat
      pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16)
#      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/easynegative.safetensors", token="easynegative")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/FastNegativeV2.pt", token="FastNegativeV2")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/bad_prompt_version2.pt", token="bad_prompt_version2")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/negative_hand-neg.pt", token="negative_hand-neg")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/amnmdl_brtsh-2850.pt", token="amnmdl_brtsh-2850")
      pipe.load_lora_weights("/home/suchitra/kellogs/embeddings/LORA/", weight_name="british_short_hair_cat.safetensors")
#      pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
      pipe.scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True,
                                                    algorithm_type="sde-dpmsolver++"
                                                    )
      pipe = pipe.to(device)
      # pipe.enable_attention_slicing()
      if selected_option == 'Feeding Experience':
        prompt = prompt + 'proportionate legs'
      prompt = prompt.replace('British Shorthair Cat', '((British Shorthair Cat))') + "<lora:british_short_hair_cat:1.2>" + ", single cat,(high detailed face:1.2),(masterpiece, finely detailed beautiful eyes: 1.2), dslr, 8k, 4k, ultrarealistic, natural skin, textured skin,  hdr, realistic, high definition"
      
      
      if selected_option == 'Feeding Experience':
        negative_prompt = "dark photos, FastNegativeV2, bad_prompt_version2, negative_hand-neg, big bowl, short legs, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"
        
      
      negative_prompt = "dark photos,((no clothes)), FastNegativeV2, bad_prompt_version2, negative_hand-neg, easynegative, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"
      
      with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
                prompt, height=int(height), width=int(width),
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cpu
            ).images
            
    elif(animal == 'Husky'):
    #FastNegativeV2, (negative_hand-neg:1.2), husdg
      pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16)
#      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/embeddings/easynegative.safetensors", token="easynegative")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/FastNegativeV2.pt", token="FastNegativeV2")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/bad_prompt_version2.pt", token="bad_prompt_version2")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/husdg.pt", token="husdg")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/negative_hand-neg.pt", token="negative_hand-neg")
      pipe.load_lora_weights("/home/suchitra/kellogs/embeddings/LORA/", weight_name="husky.safetensors")
#      pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
      pipe.scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True,
                                                    algorithm_type="sde-dpmsolver++"
                                                    )
      pipe = pipe.to(device)
      # pipe.enable_attention_slicing()
      if selected_option == 'Feeding Experience':
        prompt = prompt + 'proportionate legs'
      prompt =  prompt.replace('Husky', '((Husky))')+"<lora:husky:1.2>" + ", single dog,(high detailed face:1.2), (masterpiece, finely detailed beautiful eyes: 1.2), (RAW photo, 8k uhd, film grain),(high detailed skin:1.2), dslr, high quality, Fujifilm XT3"
      
      
      if selected_option == 'Feeding Experience':
        negative_prompt = "dark photos,((no clothes)),FastNegativeV2, negative_hand-neg,bad_prompt_version2,((short legs)), big bowls, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"
      
      negative_prompt = "dark photos,((no clothes)), negative_hand-neg, easynegative,bad_prompt_version2, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"

      with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
                prompt, height=int(height), width=int(width),
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cpu
            ).images
    elif(animal == 'Labrador'):
    #FastNegativeV2, (negative_hand-neg:1.2), husdg
      pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16)
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/easynegative.safetensors", token="easynegative")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/FastNegativeV2.pt", token="FastNegativeV2")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/lare-800.pt", token="lare-800")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/negative_hand-neg.pt", token="negative_hand-neg")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/bad_prompt_version2.pt", token="bad_prompt_version2")
#      pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
      pipe.scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True,
                                                    algorithm_type="sde-dpmsolver++"
                                                    )
      pipe = pipe.to(device)
      # pipe.enable_attention_slicing()
      if selected_option == 'Feeding Experience':
        prompt = prompt + 'proportionate legs'
      prompt = prompt.replace('Labrador', '((Labrador))') + ", single dog, (high detailed face:1.2),(masterpiece, finely detailed beautiful eyes: 1.2), dslr, 8k, 4k, ultrarealistic, realistic, natural skin, textured skin,  hdr"
      
      
      if selected_option == 'Feeding Experience':
        negative_prompt = "lare-800" + ", dark photos, ((no clothes)), FastNegativeV2, bad_prompt_version2, negative_hand-neg,big bowl, short legs, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"
      
      negative_prompt = "lare-800" + ", dark photos, ((no clothes)), FastNegativeV2, bad_prompt_version2, negative_hand-neg, easynegative, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"

      with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
                prompt, height=int(height), width=int(width),
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cpu.manual_seed(1564295580)
            ).images
    elif(animal == 'Golden-Retreiver'):
        #FastNegativeV2, (negative_hand-neg:1.2), husdg
      pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16)
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/easynegative.safetensors", token="easynegative")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/FastNegativeV2.pt", token="FastNegativeV2")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/negative_hand-neg.pt", token="negative_hand-neg")
      pipe.load_textual_inversion("/home/suchitra/kellogs/embeddings/TI/bad_prompt_version2.pt", token="bad_prompt_version2")
#      pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
      pipe.scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True,
                                                    algorithm_type="sde-dpmsolver++"
                                                    )
      pipe = pipe.to(device)
      # pipe.enable_attention_slicing()
      if selected_option == 'Feeding Experience':
        prompt = prompt + 'proportionate legs'
      prompt = prompt.replace('Golden-Retreiver', '((Golden-Retreiver))').replace('Golden Retreiver', '((Golden Retreiver))') + ", single dog, (high detailed face:1.2),(masterpiece, finely detailed beautiful eyes: 1.2), dslr, 8k, 4k, ultrarealistic, realistic, natural skin, textured skin,  hdr, realistic, high definition"
      
      
      if selected_option == 'Feeding Experience':
        negative_prompt = "dark photos, ((no clothes)), FastNegativeV2, bad_prompt_version2, negative_hand-neg, golden bowl,big bowl, short legs, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"
      
      negative_prompt = "dark photos, ((no clothes)), FastNegativeV2, bad_prompt_version2, negative_hand-neg, easynegative, bad anatomy,(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused finger"

      with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
                prompt, height=int(height), width=int(width),
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cpu
            ).images
                

def inference1(prompt, negative_prompt=neg_prompt, num_samples=5, height=512, width=512, num_inference_steps=60, guidance_scale=7, theme='Farmland'):
    #g_cpu = torch.Generator(device='cuda')
    #print(person)
      pipe = StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None, torch_dtype=torch.float16)
      pipe.scheduler = DPMSolverMultistepScheduler(use_karras_sigmas=True,
                                                    algorithm_type="sde-dpmsolver++"
                                                    )
      pipe = pipe.to(device)
      # pipe.enable_attention_slicing()
      if theme == 'Farmland':
        prompt = 'An image of an agricultural land with tractor with realistic sky,dslr, 8k, 4k, ultrarealistic, realistic'
        negative_prompt = "cropped photo,animated, cartoon,cropped, out of frame, worst quality, low quality, jpeg artifacts, unrealistic,unrealistic machine,weird machine infornt of tractors, large vegetable, large leaf, cartoon image, anime image"
        
      elif theme == 'Poultry':
        prompt = 'An image of a poultry farm with some chickens, turkeys and chicks eating food, (highly detailed chickens with 2 legs: 1.2)'
        negative_prompt = "FastNegativeV2,4 legged chicken, multi legged chicken unrealistic, morphed chicken,mutation,bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, deformed, blurry, multiple heads, cartoon image, anime image"
        
      else:
        prompt = 'a bog food bowl containing portions of peas, carrots and salmon,dslr, 8k, 4k, ultrarealistic, realistic'
        negative_prompt = "dark photos,cropped photo,animated, cartoon,cropped, out of frame, worst quality, low quality, jpeg artifacts,unrealistic proportions ,large peas, cartoon image, anime image"
        
      prompt = prompt + "(masterpiece, sidelighting, finely detailed beautiful: 1.2), dslr, 8k, 4k, ultrarealistic, realistic, hdr, realistic, high definition"
      
      

      with torch.autocast("cuda"), torch.inference_mode():
        return pipe(
                prompt, height=int(height), width=int(width),
                negative_prompt=negative_prompt,
                num_images_per_prompt=int(num_samples),
                num_inference_steps=int(num_inference_steps), guidance_scale=guidance_scale,
                generator=g_cpu
            ).images
      