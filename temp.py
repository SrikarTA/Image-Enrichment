import os

path = '/home/suchitra/kellogs/Outputs/Image/'
img_paths = os.listdir('/home/suchitra/kellogs/Outputs/Image/')

for img in img_paths:
  print(path+img)