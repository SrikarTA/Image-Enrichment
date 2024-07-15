import os

import cv2

from config import *


def get_video(image_folder,output1=output_video,output2=final_video):
    # Get a list of image file names in the folder
    image_files = sorted(os.listdir(image_folder))

    # Get the first image dimensions to set up the video writer
    first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
    height, width, _ = first_image.shape

    # Define the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))

    # Iterate through each image and write it to the video
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        video_writer.write(image)

        # Add a delay of 2.5 seconds (2500 milliseconds)
        delay = int(30 * 2.5)  # 30 frames per second
        for _ in range(delay):
            video_writer.write(image)

    # Release the video writer
    video_writer.release()

    # # Run ffmpeg command to add audio to the video
    audio_cmd = f"ffmpeg -i {output1} -i {audio_path} -map 0:v -map 1:a -c:v copy -c:a aac -shortest -y {output2}"
    os.system(audio_cmd)

