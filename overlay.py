import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from config import * 

# BACKGROUND_IMAGE_PATH = r"kid_image.png"
# OVERLAY_IMAGE_PATH = r"box_image.jpg"
# img1 = Image.open(BACKGROUND_IMAGE_PATH)
# img2 = Image.open(OVERLAY_IMAGE_PATH)

# loc=img1.size[0]-img2.size[0]
# img1.paste(img2, (0,loc))

# img1.show()

def overlay_img(background_img_folder,product_selected):
    image_path = image_folder
    bg_imgs = os.listdir(image_path)
    if product_selected == "cereals":
       product_path = f"{overlay_path}/cereals.png"
       
    else:
        product_path = f"{overlay_path}/chocos.png"

    overlayimg = Image.open(product_path).resize((120,180)) 
    print(overlayimg.size)

    for img in bg_imgs:
        im = os.path.join(image_path,img)
        img1 = Image.open(im)
        loc=img1.size[0]-overlayimg.size[0]
        img1.paste(overlayimg, (0,loc))
        img1.save(f"{out_path}/"+img)


