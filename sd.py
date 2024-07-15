from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from app import *

model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)


# pipe.load_textual_inversion("/home/suchitra/a1111/stable-diffusion-webui/embeddings/FastNegativeV2.pt", token="FastNegativeV2")
pipe.load_textual_inversion("/home/suchitra/a1111/stable-diffusion-webui/embeddings/bad_prompt_version2.pt", token="bad_prompt_version2")
pipe.load_textual_inversion("/home/suchitra/a1111/stable-diffusion-webui/embeddings/negative_hand-neg.pt", token="negative_hand-neg")


TI = {
    "Persian Cat": "pxrc-3200",
    "American Shorthair Cat": "amersgrtsq-1900",
    "British Shorthair Cat": "amnmdl_brtsh-2850",
    "Husky": "husdg-3200",
    "Labrador": "lare",
    "Golden-Retreiver": "gore-2050"
}

LORA = {
      "Persian Cat": "pxrc-3200",
      "American Shorthair Cat": "amersgrtsq-1900",
      "British Shorthair Cat": "amnmdl_brtsh-2850",
      "Husky": "husdg-3200"
}


if option3 in TI and option3 in LORA:
  pipe.load_textual_inversion(f"/home/suchitra/a1111/stable-diffusion-webui/embeddings/{TI[option3]}.pt", token=f"{TI[option3]}")
  pipe.load_lora_weights("/home/suchitra/a1111/stable-diffusion-webui/models/LoRA/", weight_name="{LORA[option3]}.safetensors")
elif option3 in TI:
  pipe.load_textual_inversion(f"/home/suchitra/a1111/stable-diffusion-webui/embeddings/{TI[option3]}.pt", token=f"{TI[option3]}")
elif option3 in LORA:
  pipe.load_lora_weights("/home/suchitra/a1111/stable-diffusion-webui/models/LoRA/", weight_name="{LORA[option3]}.safetensors")
else:
  pipe.load_lora_weights(".", weight_name="olsen.safetensors")


if option3 in TI and option3 in LORA:
  prompt = TI[option3] + response + f"<lora:{LORA[option3]}:1.2>"+", ultra realistic, UHD, (RAW photo, 8k uhd, film grain),(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT4"
elif option3 in TI:
  prompt = TI[option3] + response +", ultra realistic, UHD, (RAW photo, 8k uhd, film grain),(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT4"
elif option3 in LORA:
  prompt = response + f"<lora:{LORA[option3]}:1.2>"+", ultra realistic, UHD, (RAW photo, 8k uhd, film grain),(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT4"
else:
  prompt =  response + f"<lora:olsen:1.2>"+,"ultra realistic, UHD, (RAW photo, 8k uhd, film grain),(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT4"
  
# prompt = "A <lora:olsen:1.2> women olsen standing in a red carpet, ultra realistic, UHD, (RAW photo, 8k uhd, film grain),(high detailed skin:1.2), 8k uhd, dslr, soft lighting, high quality, film grain, Fujifilm XT4"

negative_prompt = "FastNegativeV2, bad_prompt_version2, big cat"

image = pipe(prompt=prompt,negative_prompt=negative_prompt, num_inference_steps=50).images[0]

image.save("olsen.png")

import subprocess

command = "python /home/suchitra/CodeFormer/inference_codeformer.py -w 0.7 --input_path /home/suchitra/kellogs --bg_upsampler realesrgan --face_upsample --output_path /home/suchitra/kellogs/output"

# Run the command
subprocess.run(command, shell=True)